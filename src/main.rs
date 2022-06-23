// #![allow(dead_code)]
// #![allow(unused_variables)]
// #![allow(unreachable_code)]
use ash::extensions::khr;
use ash::vk;
use std::ffi::*;
use std::mem::ManuallyDrop;
use winit::event::{Event, WindowEvent};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let event_loop = winit::event_loop::EventLoop::new();
    let window = winit::window::WindowBuilder::new()
        .with_title("Lyra-Vulkan")
        .with_inner_size(winit::dpi::LogicalSize::new(512.0, 512.0))
        .build(&event_loop)
        .unwrap();
    let mut lyra = VulkanSt::init(window)?;
    event_loop.run(move |event, _, controlflow| match event {
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => {
            *controlflow = winit::event_loop::ControlFlow::Exit;
        }
        Event::MainEventsCleared => {
            // do work here later
            lyra.window.request_redraw();
        }
        Event::RedrawRequested(_) => {
            // Aquire the next image
            lyra.swapchain.current_image =
                (lyra.swapchain.current_image + 1) % lyra.swapchain.amount_of_images as usize;

            let (image_index, _) = unsafe {
                lyra.swapchain
                    .swapchain_loader
                    .acquire_next_image(
                        lyra.swapchain.swapchain,
                        std::u64::MAX,
                        lyra.swapchain.image_available[lyra.swapchain.current_image],
                        vk::Fence::null(),
                    )
                    .expect("Issue with image acquisition")
            };
            unsafe {
                lyra.device
                    .wait_for_fences(
                        &[lyra.swapchain.may_begin_drawing[lyra.swapchain.current_image]],
                        true,
                        std::u64::MAX,
                    )
                    .expect("Fences Waiting")
            };
            unsafe {
                lyra.device
                    .reset_fences(&[lyra.swapchain.may_begin_drawing[lyra.swapchain.current_image]])
                    .expect("Resetting Fences")
            }

            let semaphores_available =
                [lyra.swapchain.image_available[lyra.swapchain.current_image]];
            let waiting_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let semaphores_finished =
                [lyra.swapchain.rendering_finished[lyra.swapchain.current_image]];
            let commandbuffers = [lyra.commandbuffers[image_index as usize]];
            let submit_info = [vk::SubmitInfo::builder()
                .wait_semaphores(&semaphores_available)
                .wait_dst_stage_mask(&waiting_stages)
                .command_buffers(&commandbuffers)
                .signal_semaphores(&semaphores_finished)
                .build()];

            unsafe {
                lyra.device
                    .queue_submit(
                        lyra.queues.graphics_queue,
                        &submit_info,
                        lyra.swapchain.may_begin_drawing[lyra.swapchain.current_image],
                    )
                    .expect("Queue Submission")
            }
            let swapchains = [lyra.swapchain.swapchain];
            let indices = [image_index];
            let present_info = vk::PresentInfoKHR::builder()
                .wait_semaphores(&semaphores_finished)
                .swapchains(&swapchains)
                .image_indices(&indices);
            unsafe {
                lyra.swapchain
                    .swapchain_loader
                    .queue_present(lyra.queues.graphics_queue, &present_info)
                    .expect("Queue Presentation")
            };
        }
        _ => {}
    });
    Ok(())
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
        let debug_create_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
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
    framebuffers: Vec<vk::Framebuffer>,
    surface_format: vk::SurfaceFormatKHR,
    extent: vk::Extent2D,
    image_available: Vec<vk::Semaphore>,
    rendering_finished: Vec<vk::Semaphore>,
    may_begin_drawing: Vec<vk::Fence>,
    amount_of_images: u32,
    current_image: usize,
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
        let extent = surface_capabilities.current_extent;
        let surface_present_modes = surfaces.get_present_modes(physical_device)?;
        let surface_format = *surfaces.get_formats(physical_device)?.first().unwrap();
        let queuefamilies = [queue_families.graphics_q_index.unwrap()];
        let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
            .surface(surfaces.surface)
            .min_image_count(
                3.max(surface_capabilities.min_image_count)
                    .min(surface_capabilities.max_image_count),
            )
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
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
        let amount_of_images = swapchain_images.len() as u32;
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
        let mut image_available = vec![];
        let mut rendering_finished = vec![];
        let mut may_begin_drawing = vec![];
        let semaphoreinfo = vk::SemaphoreCreateInfo::builder();
        let fenceinfo = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);
        for _ in 0..amount_of_images {
            let semaphore_available =
                unsafe { logical_device.create_semaphore(&semaphoreinfo, None) }?;
            let semaphore_finished =
                unsafe { logical_device.create_semaphore(&semaphoreinfo, None) }?;
            image_available.push(semaphore_available);
            rendering_finished.push(semaphore_finished);
            let fence = unsafe { logical_device.create_fence(&fenceinfo, None) }?;
            may_begin_drawing.push(fence);
        }
        Ok(SwapchainSt {
            swapchain_loader,
            swapchain,
            images: swapchain_images,
            imageviews: swapchain_imageviews,
            framebuffers: vec![],
            surface_format,
            extent,
            image_available,
            rendering_finished,
            may_begin_drawing,
            amount_of_images,
            current_image: 0,
        })
    }
    fn create_framebuffers(
        &mut self,
        logical_device: &ash::Device,
        renderpass: vk::RenderPass,
    ) -> Result<(), vk::Result> {
        for iv in &self.imageviews {
            let iview = [*iv];
            let framebuffer_info = vk::FramebufferCreateInfo::builder()
                .render_pass(renderpass)
                .attachments(&iview)
                .width(self.extent.width)
                .height(self.extent.height)
                .layers(1);
            let fb = unsafe { logical_device.create_framebuffer(&framebuffer_info, None) }?;
            self.framebuffers.push(fb);
        }
        Ok(())
    }
    unsafe fn cleanup(&mut self, logical_device: &ash::Device) {
        for fence in &self.may_begin_drawing {
            logical_device.destroy_fence(*fence, None);
        }
        for semaphore in &self.image_available {
            logical_device.destroy_semaphore(*semaphore, None);
        }
        for semaphore in &self.rendering_finished {
            logical_device.destroy_semaphore(*semaphore, None);
        }
        for fb in &self.framebuffers {
            logical_device.destroy_framebuffer(*fb, None);
        }
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
    format: vk::Format,
) -> Result<vk::RenderPass, vk::Result> {
    let attachments = [vk::AttachmentDescription::builder()
        .format(format)
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

struct PipelineSt {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
}

impl PipelineSt {
    fn init(
        logical_device: &ash::Device,
        swapchain: &SwapchainSt,
        renderpass: &vk::RenderPass,
    ) -> Result<PipelineSt, vk::Result> {
        let vertexshader_create_info = vk::ShaderModuleCreateInfo::builder()
            .code(vk_shader_macros::include_glsl!("./shaders/shader.vert"));
        let vertexshader_module =
            unsafe { logical_device.create_shader_module(&vertexshader_create_info, None)? };

        let fragmentshader_create_info = vk::ShaderModuleCreateInfo::builder()
            .code(vk_shader_macros::include_glsl!("./shaders/shader.frag"));
        let fragmentshader_module =
            unsafe { logical_device.create_shader_module(&fragmentshader_create_info, None)? };

        let mainfunctionname = CString::new("main").unwrap();
        let vertexshader_stage = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vertexshader_module)
            .name(&mainfunctionname);
        let fragmentshader_stage = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fragmentshader_module)
            .name(&mainfunctionname);
        let shader_stages = vec![vertexshader_stage.build(), fragmentshader_stage.build()];

        let vertex_attrib_descs = [vk::VertexInputAttributeDescription {
            binding: 0,
            location: 0,
            offset: 0,
            format: vk::Format::R32G32B32A32_SFLOAT,
        }];
        let vertex_binding_descs = [vk::VertexInputBindingDescription {
            binding: 0,
            stride: 16,
            input_rate: vk::VertexInputRate::VERTEX,
        }];
        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_attribute_descriptions(&vertex_attrib_descs)
            .vertex_binding_descriptions(&vertex_binding_descs);

        let input_assembly_info = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::POINT_LIST);

        let viewports = [vk::Viewport {
            x: 0.,
            y: 0.,
            width: swapchain.extent.width as f32,
            height: swapchain.extent.height as f32,
            min_depth: 0.,
            max_depth: 1.,
        }];

        let scissors = [vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: swapchain.extent,
        }];

        let viewport_info = vk::PipelineViewportStateCreateInfo::builder()
            .viewports(&viewports)
            .scissors(&scissors);

        let rasterizer_info = vk::PipelineRasterizationStateCreateInfo::builder()
            .line_width(1.0)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .cull_mode(vk::CullModeFlags::NONE)
            .polygon_mode(vk::PolygonMode::FILL);

        let multisample_info = vk::PipelineMultisampleStateCreateInfo::builder()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1); // 1spp

        let colorblend_attachments = [vk::PipelineColorBlendAttachmentState::builder()
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_write_mask(
                vk::ColorComponentFlags::R
                    | vk::ColorComponentFlags::G
                    | vk::ColorComponentFlags::B
                    | vk::ColorComponentFlags::A,
            )
            .build()];
        let colorblend_info =
            vk::PipelineColorBlendStateCreateInfo::builder().attachments(&colorblend_attachments);

        let pipelinelayout_info = vk::PipelineLayoutCreateInfo::builder();
        let pipelinelayout =
            unsafe { logical_device.create_pipeline_layout(&pipelinelayout_info, None)? };

        let pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&input_assembly_info)
            .viewport_state(&viewport_info)
            .rasterization_state(&rasterizer_info)
            .multisample_state(&multisample_info)
            .color_blend_state(&colorblend_info)
            .layout(pipelinelayout)
            .render_pass(*renderpass)
            .subpass(0);
        let graphics_pipeline = unsafe {
            logical_device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    &[pipeline_info.build()],
                    None,
                )
                .expect("Problem with Pipeline creation")
        }[0];

        unsafe {
            // We can destroy the shader modules as soon as we finish creating the pipeline
            logical_device.destroy_shader_module(fragmentshader_module, None);
            logical_device.destroy_shader_module(vertexshader_module, None);
        }
        Ok(PipelineSt {
            pipeline: graphics_pipeline,
            layout: pipelinelayout,
        })
    }
    fn cleanup(&self, logical_device: &ash::Device) {
        unsafe {
            logical_device.destroy_pipeline(self.pipeline, None);
            logical_device.destroy_pipeline_layout(self.layout, None);
        }
    }
}

struct PoolsSt {
    commandpool_graphics: vk::CommandPool,
    commandpool_transfer: vk::CommandPool,
}

impl PoolsSt {
    fn init(
        logical_device: &ash::Device,
        queue_families: &QueueFamilies,
    ) -> Result<PoolsSt, vk::Result> {
        let graphics_commandpool_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(queue_families.graphics_q_index.unwrap())
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER); // We can reuse the command buffers
        let commandpool_graphics =
            unsafe { logical_device.create_command_pool(&graphics_commandpool_info, None)? };
        let transfer_commandpool_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(queue_families.transfer_q_index.unwrap())
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let commandpool_transfer =
            unsafe { logical_device.create_command_pool(&transfer_commandpool_info, None)? };

        Ok(PoolsSt {
            commandpool_graphics,
            commandpool_transfer,
        })
    }
    fn cleanup(&self, logical_device: &ash::Device) {
        unsafe {
            logical_device.destroy_command_pool(self.commandpool_graphics, None);
            logical_device.destroy_command_pool(self.commandpool_transfer, None);
        }
    }
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
    pipeline: PipelineSt,
    pools: PoolsSt,
    commandbuffers: Vec<vk::CommandBuffer>,
}

fn create_commandbuffers(
    logical_device: &ash::Device,
    pools: &PoolsSt,
    amount: usize,
) -> Result<Vec<vk::CommandBuffer>, vk::Result> {
    let commandbuffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
        .command_pool(pools.commandpool_graphics)
        .command_buffer_count(amount as u32);
    unsafe { logical_device.allocate_command_buffers(&commandbuffer_allocate_info) }
}

fn fill_commandbuffers(
    commandbuffers: &[vk::CommandBuffer],
    logical_device: &ash::Device,
    renderpass: &vk::RenderPass,
    swapchain: &SwapchainSt,
    pipeline: &PipelineSt,
) -> Result<(), vk::Result> {
    for (i, &commandbuffer) in commandbuffers.iter().enumerate() {
        let commandbuffer_begininfo = vk::CommandBufferBeginInfo::builder();
        unsafe { logical_device.begin_command_buffer(commandbuffer, &commandbuffer_begininfo)? };

        let clearvalues = [vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.1, 0.1, 0.2, 1.0],
            },
        }];
        let renderpass_begininfo = vk::RenderPassBeginInfo::builder()
            .render_pass(*renderpass)
            .framebuffer(swapchain.framebuffers[i])
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: swapchain.extent,
            })
            .clear_values(&clearvalues);

        unsafe {
            logical_device.cmd_begin_render_pass(
                commandbuffer,
                &renderpass_begininfo,
                vk::SubpassContents::INLINE,
            );

            logical_device.cmd_bind_pipeline(
                commandbuffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.pipeline,
            );
            logical_device.cmd_draw(commandbuffer, 1, 1, 0, 0);
            logical_device.cmd_end_render_pass(commandbuffer);
            logical_device.end_command_buffer(commandbuffer)?;
        }
    }
    Ok(())
}

impl VulkanSt {
    fn init(window: winit::window::Window) -> Result<VulkanSt, Box<dyn std::error::Error>> {
        // Loads the Vulkan library and unwraps the result((?))-, erroring if it fails and ending the program
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

        let mut swapchain = SwapchainSt::init(
            &instance,
            physical_device,
            &logical_device,
            &surfaces,
            &queue_families,
            &queues,
        )?;

        let renderpass = init_renderpass(
            &logical_device,
            physical_device,
            swapchain.surface_format.format,
        )?;
        swapchain.create_framebuffers(&logical_device, renderpass)?;

        let pipeline = PipelineSt::init(&logical_device, &swapchain, &renderpass)?;

        let pools = PoolsSt::init(&logical_device, &queue_families)?;

        let commandbuffers =
            create_commandbuffers(&logical_device, &pools, swapchain.framebuffers.len())?;

        fill_commandbuffers(
            &commandbuffers,
            &logical_device,
            &renderpass,
            &swapchain,
            &pipeline,
        )?;

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
            pipeline,
            pools,
            commandbuffers,
        })
    }
}

// TODO: Get rid of all the ManuallyDrop::drop garbage
impl Drop for VulkanSt {
    fn drop(&mut self) {
        unsafe {
            self.device
                .device_wait_idle()
                .expect("Something went wrong while waiting");
            self.pools.cleanup(&self.device);
            self.pipeline.cleanup(&self.device);
            self.device.destroy_render_pass(self.renderpass, None);
            self.swapchain.cleanup(&self.device);
            self.device.destroy_device(None);
            std::mem::ManuallyDrop::drop(&mut self.surfaces);
            std::mem::ManuallyDrop::drop(&mut self.debug);
            self.instance.destroy_instance(None);
        };
    }
}
// Validation layers will put their messages here. 'extern' function callback to some external C code
unsafe extern "system" fn vulkan_debug_utils_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT, // Returns a raw pointer to debug message, needs deref
    _p_user_data: *mut c_void, // Pointer of an unspecified type, we don't use this
) -> vk::Bool32 {
    let message = CStr::from_ptr((*p_callback_data).p_message);
    let severity = format!("{:?}", message_severity).to_lowercase();
    let mtype = format!("{:?}", message_type).to_lowercase();
    println!("[Debug][{}][{}]{:?}", severity, mtype, message);
    vk::FALSE // Should we skip the call to the driver?
}
