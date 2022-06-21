use ash::extensions::khr;
use ash::vk;
use std::ffi::*;
use std::mem::ManuallyDrop;
use winit::event::{Event, WindowEvent};
// use ash::vk::EntryFnV1_3;
// use ash::vk::DeviceFnV1_3;
// use ash::vk::InstanceFnV1_3;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let event_loop = winit::event_loop::EventLoop::new();
    let window = winit::window::WindowBuilder::new()
        .with_title("Lyra-Vulkan")
        .with_inner_size(winit::dpi::LogicalSize::new(1024.0, 1024.0))
        .build(&event_loop)
        .unwrap();
    let vulkanloader = VulkanSt::init(window)?;
    event_loop.run(move |event, _, controlflow| match event {
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => {
            *controlflow = winit::event_loop::ControlFlow::Exit;
        }
        _ => {
            // do work here later
            vulkanloader.window.request_redraw();
        }
        Event::RedrawRequested(_) => {
            // render here later
        }
    });
    Ok(())
}

struct VulkanSt {
    window: winit::window::Window,
    entry: ash::Entry,
    instance: ash::Instance,
    debug: ManuallyDrop<DebugSt>,
    surfaces: ManuallyDrop<SurfaceSt>,
    physical_device: vk::PhysicalDevice,
    physical_device_properties: vk::PhysicalDeviceProperties,
    queue_families: QueueFamilies,
    queues: Queues,
    device: ash::Device,
    swapchain: SwapchainSt,
    renderpass: vk::RenderPass,
}

impl VulkanSt {
    fn init(window: winit::window::Window) -> Result<VulkanSt, Box<dyn std::error::Error>> {
        // Loads the vulkan library and unwraps the result((?))-, erroring if it fails and ending the program
        let entry = unsafe { ash::Entry::load()? };

        let layer_names = vec![CString::new("VK_LAYER_KHRONOS_validation").unwrap()];
        let instance = init_instance(&entry, &layer_names)?;
        let debug = DebugSt::init(&entry, &instance)?;
        // Create our window surface and surface loader (an entry to surface-related functions)
        let surfaces = SurfaceSt::init(&window, &entry, &instance)?;

        let (physical_device, physical_device_properties) =
            init_physical_device_and_properties(&instance)?;

        let queue_families = QueueFamilies::init(&instance, physical_device, &surfaces)?;

        let (logical_device, queues) =
            init_device_and_queues(&instance, physical_device, &queue_families, &layer_names)?;

        let swapchain = SwapchainSt::init(
            &instance,
            physical_device,
            &logical_device,
            &surfaces,
            &queue_families,
            &queues,
        )?;

        let renderpass = init_renderpass(&logical_device, physical_device, &surfaces)?;

        Ok(VulkanSt {
            window,
            entry,
            instance,
            debug: ManuallyDrop::new(debug),
            surfaces: ManuallyDrop::new(surfaces),
            physical_device,
            physical_device_properties,
            queue_families,
            queues,
            device: logical_device,
            swapchain,
            renderpass,
        })
    }
}

// TODO: Get rid of all the ManuallyDrop::drop garbage
impl Drop for VulkanSt {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_render_pass(self.renderpass, None);
            self.swapchain.cleanup(&self.device);
            self.device.destroy_device(None);
            std::mem::ManuallyDrop::drop(&mut self.surfaces);
            std::mem::ManuallyDrop::drop(&mut self.debug);
            self.instance.destroy_instance(None);
        };
    }
}

fn layer_names_as_ptrs(layer_names: &Vec<CString>) -> Vec<*const i8> {
    layer_names.iter().map(|layers| layers.as_ptr()).collect()
}

fn init_instance(
    entry: &ash::Entry,
    layer_names: &Vec<CString>,
) -> Result<ash::Instance, ash::vk::Result> {
    let enginename = CString::new("RenderDev").unwrap();
    let appname = CString::new("Lyra").unwrap();
    let layer_names_ptrs = layer_names_as_ptrs(&layer_names);

    let mut debug_create_info = vk::DebugUtilsMessengerCreateInfoEXT::builder() // Duplicate code from the DebugST impl but not sure how to fix
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
    // | vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
    // | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
    | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
        )
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
        )
        .pfn_user_callback(Some(vulkan_debug_utils_callback));

    let app_info = vk::ApplicationInfo::builder()
        .application_name(&appname)
        .application_version(vk::make_api_version(0, 0, 0, 1))
        .engine_name(&enginename)
        .engine_version(vk::make_api_version(0, 0, 42, 0))
        .application_version(vk::make_api_version(0, 0, 0, 1))
        .api_version(vk::make_api_version(0, 1, 0, 106));

    let extension_name_pointers: Vec<*const i8> = // TODO Can possbily break things, wouldn't accept i8 for some reason. Could cause cross-platform issues.
        vec![
            ash::extensions::ext::DebugUtils::name().as_ptr(),
            ash::extensions::ext::DebugUtils::name().as_ptr(),
            ash::extensions::khr::Surface::name().as_ptr(),
            ash::extensions::khr::Win32Surface::name().as_ptr()];

    let instance_create_info = vk::InstanceCreateInfo::builder()
        .push_next(&mut debug_create_info) // We don't need this but we can get notified on errors with instance creation if we do.
        .application_info(&app_info)
        .enabled_layer_names(&layer_names_ptrs)
        .enabled_extension_names(&extension_name_pointers);
    unsafe { entry.create_instance(&instance_create_info, None) }
}

struct DebugSt {
    loader: ash::extensions::ext::DebugUtils,
    messenger: vk::DebugUtilsMessengerEXT,
}
impl DebugSt {
    fn init(entry: &ash::Entry, instance: &ash::Instance) -> Result<DebugSt, vk::Result> {
        let mut debug_create_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
        // | vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
        // | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
        | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
            )
            .pfn_user_callback(Some(vulkan_debug_utils_callback));
        let loader = ash::extensions::ext::DebugUtils::new(entry, instance);
        let messenger = unsafe { loader.create_debug_utils_messenger(&debug_create_info, None)? };

        Ok(Self { loader, messenger })
    }
}
impl Drop for DebugSt {
    fn drop(&mut self) {
        unsafe {
            self.loader
                .destroy_debug_utils_messenger(self.messenger, None)
        };
    }
}

struct SurfaceSt {
    surface: vk::SurfaceKHR,
    surface_loader: khr::Surface,
}
impl SurfaceSt {
    fn init(
        window: &winit::window::Window,
        entry: &ash::Entry,
        instance: &ash::Instance,
    ) -> Result<SurfaceSt, vk::Result> {
        let surface =
            unsafe { ash_window::create_surface(&entry, &instance, &window, None).unwrap() };
        let surface_loader = khr::Surface::new(&entry, &instance);
        Ok(SurfaceSt {
            surface,
            surface_loader,
        })
    }
    fn get_capabilities(
        &self,
        physical_device: vk::PhysicalDevice,
    ) -> Result<vk::SurfaceCapabilitiesKHR, vk::Result> {
        unsafe {
            self.surface_loader
                .get_physical_device_surface_capabilities(physical_device, self.surface)
        }
    }
    fn get_present_modes(
        &self,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Vec<vk::PresentModeKHR>, vk::Result> {
        unsafe {
            self.surface_loader
                .get_physical_device_surface_present_modes(physical_device, self.surface)
        }
    }
    fn get_formats(
        &self,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Vec<vk::SurfaceFormatKHR>, vk::Result> {
        unsafe {
            self.surface_loader
                .get_physical_device_surface_formats(physical_device, self.surface)
        }
    }
    fn get_physical_device_surface_support(
        &self,
        physical_device: vk::PhysicalDevice,
        queue_family_index: usize,
    ) -> Result<bool, vk::Result> {
        unsafe {
            self.surface_loader.get_physical_device_surface_support(
                physical_device,
                queue_family_index as u32,
                self.surface,
            )
        }
    }
}
impl Drop for SurfaceSt {
    fn drop(&mut self) {
        unsafe {
            self.surface_loader.destroy_surface(self.surface, None);
        }
    }
}

// Queues are where we submit commands to; in some sense they are the place where our graphics commands stand in a queue and wait.
// Queue families are collections of such queues, grouped by their “specialization”. They may be made for graphics commands, or for transfer operations, or for computing.
// Which queue families are available is a property of the physical device (and this might also play a role in selection of said device).
// We choose one capable of graphics and one suitable for transfer operations.
// For transfer we prefer a queue family that is dedicated to transfer only, hoping that we thereby hit specialised hardware which is faster.
// But we could also just use the same.

// This comes with the warning that (theoretically)
// it is possible that the (only) queue available for presenting on the surface and the (only) queues capable of “GRAPHICS” drawing differ.
// This is not the case here and this is learning code so we’ll ignore this potential problem.
struct QueueFamilies {
    graphics_q_index: Option<u32>,
    transfer_q_index: Option<u32>,
}
impl QueueFamilies {
    fn init(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        surfaces: &SurfaceSt,
    ) -> Result<QueueFamilies, vk::Result> {
        let queuefamilyproperties =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        let mut found_graphics_q_index = None;
        let mut found_transfer_q_index = None;
        for (index, qfam) in queuefamilyproperties.iter().enumerate() {
            if qfam.queue_count > 0
                && qfam.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                && surfaces.get_physical_device_surface_support(physical_device, index)?
            {
                found_graphics_q_index = Some(index as u32);
            }
            if qfam.queue_count > 0 && qfam.queue_flags.contains(vk::QueueFlags::TRANSFER) {
                if found_transfer_q_index.is_none()
                    || !qfam.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                {
                    found_transfer_q_index = Some(index as u32);
                }
            }
        }
        Ok(QueueFamilies {
            graphics_q_index: found_graphics_q_index,
            transfer_q_index: found_transfer_q_index,
        })
    }
}

struct SwapchainSt {
    swapchain_loader: ash::extensions::khr::Swapchain,
    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    imageviews: Vec<vk::ImageView>,
}

impl SwapchainSt {
    fn init(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        logical_device: &ash::Device,
        surfaces: &SurfaceSt,
        queue_families: &QueueFamilies,
        queues: &Queues,
    ) -> Result<SwapchainSt, vk::Result> {
        let surface_capabilities = surfaces.get_capabilities(physical_device)?;
        let surface_present_modes = surfaces.get_present_modes(physical_device)?;
        let surface_formats = surfaces.get_formats(physical_device)?;
        let queuefamilies = [queue_families.graphics_q_index.unwrap()];
        let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
            .surface(surfaces.surface)
            .min_image_count(
                3.max(surface_capabilities.min_image_count)
                    .min(surface_capabilities.max_image_count),
            )
            .image_format(surface_formats.first().unwrap().format)
            .image_color_space(surface_formats.first().unwrap().color_space)
            .image_extent(surface_capabilities.current_extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .queue_family_indices(&queuefamilies)
            .pre_transform(surface_capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(vk::PresentModeKHR::IMMEDIATE);
        let swapchain_loader = ash::extensions::khr::Swapchain::new(instance, logical_device);
        let swapchain = unsafe { swapchain_loader.create_swapchain(&swapchain_create_info, None)? };
        let swapchain_images = unsafe { swapchain_loader.get_swapchain_images(swapchain)? };
        let mut swapchain_imageviews = Vec::with_capacity(swapchain_images.len());
        for image in &swapchain_images {
            let subresource_range = vk::ImageSubresourceRange::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1);
            let imageview_create_info = vk::ImageViewCreateInfo::builder()
                .image(*image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(vk::Format::B8G8R8A8_UNORM)
                .subresource_range(*subresource_range);
            let imageview =
                unsafe { logical_device.create_image_view(&imageview_create_info, None) }?;
            swapchain_imageviews.push(imageview);
        }
        Ok(SwapchainSt {
            swapchain_loader,
            swapchain,
            images: swapchain_images,
            imageviews: swapchain_imageviews,
        })
    }
    unsafe fn cleanup(&mut self, logical_device: &ash::Device) {
        for iv in &self.imageviews {
            logical_device.destroy_image_view(*iv, None);
        }
        self.swapchain_loader
            .destroy_swapchain(self.swapchain, None);
    }
}

struct Queues {
    graphics_queue: vk::Queue,
    transfer_queue: vk::Queue,
}

fn init_device_and_queues(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    queue_families: &QueueFamilies,
    layer_names: &Vec<CString>,
) -> Result<(ash::Device, Queues), vk::Result> {
    let layer_names_pointers = layer_names_as_ptrs(&layer_names);

    let priorities = [1.0f32];
    let queue_infos = [
        vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(queue_families.graphics_q_index.unwrap())
            .queue_priorities(&priorities)
            .build(),
        vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(queue_families.transfer_q_index.unwrap())
            .queue_priorities(&priorities)
            .build(),
    ];

    let device_extension_name_pointers: Vec<*const i8> =
        vec![ash::extensions::khr::Swapchain::name().as_ptr()];

    let device_create_info = vk::DeviceCreateInfo::builder()
        .queue_create_infos(&queue_infos)
        .enabled_layer_names(&layer_names_pointers) // Technically unimportant and ignored by recent implementations; maybe useful for compatability reasons
        .enabled_extension_names(&device_extension_name_pointers);

    let logical_device = unsafe {
        instance
            .create_device(physical_device, &device_create_info, None)
            .unwrap()
    };
    let graphics_queue =
        unsafe { logical_device.get_device_queue(queue_families.graphics_q_index.unwrap(), 0) };
    let transfer_queue =
        unsafe { logical_device.get_device_queue(queue_families.transfer_q_index.unwrap(), 0) };

    Ok((
        logical_device,
        Queues {
            graphics_queue,
            transfer_queue,
        },
    ))
}

fn init_physical_device_and_properties(
    instance: &ash::Instance,
) -> Result<(vk::PhysicalDevice, vk::PhysicalDeviceProperties), vk::Result> {
    // Chooses the physical device to render to. TODO make this more sensible later.
    let phys_devices = unsafe { instance.enumerate_physical_devices()? };
    let mut chosen = None;
    for p in phys_devices {
        let properties = unsafe { instance.get_physical_device_properties(p) };
        if properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
            chosen = Some((p, properties));
        }
    }
    Ok(chosen.unwrap())
}

fn init_renderpass(
    logical_device: &ash::Device,
    physical_device: vk::PhysicalDevice,
    surfaces: &SurfaceSt,
) -> Result<vk::RenderPass, vk::Result> {
    let attachments = [vk::AttachmentDescription::builder()
        .format(
            surfaces
                .get_formats(physical_device)?
                .first()
                .unwrap()
                .format,
        )
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
        .samples(vk::SampleCountFlags::TYPE_1)
        .build()];
    let color_attachment_references = [vk::AttachmentReference {
        attachment: 0,
        layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    }];

    let subpasses = [vk::SubpassDescription::builder()
        .color_attachments(&color_attachment_references)
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .build()];
    let subpass_dependencies = [vk::SubpassDependency::builder()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_subpass(0)
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_access_mask(
            vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        )
        .build()];

    let renderpass_info = vk::RenderPassCreateInfo::builder()
        .attachments(&attachments)
        .subpasses(&subpasses)
        .dependencies(&subpass_dependencies);
    let renderpass = unsafe { logical_device.create_render_pass(&renderpass_info, None)? };
    Ok(renderpass)
}

// Validation layers will put their messages here. 'extern' function callback to some external C code
unsafe extern "system" fn vulkan_debug_utils_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT, // Returns a raw pointer to the debug message which is why we deref here
    _p_user_data: *mut c_void, // Pointer of an unspecified type, we don't use this
) -> vk::Bool32 {
    let message = CStr::from_ptr((*p_callback_data).p_message);
    let severity = format!("{:?}", message_severity).to_lowercase();
    let mtype = format!("{:?}", message_type).to_lowercase();
    println!("[Debug][{}][{}]{:?}", severity, mtype, message);
    vk::FALSE // Should we skip the call to the driver?
}
