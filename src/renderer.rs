use ash::vk;

use std::cell::RefCell;
use std::ffi::CString;
use std::time::{Instant, Duration};
use std::rc::Rc;
use std::ptr;
use std::os::raw::c_void;
use std::pin::Pin;

use super::{Device, InnerDevice, Queue, FrameId, PerFrameSet};
use super::image::{Image, ImageView, ImageBuilder};
use super::texture::Texture;
use super::shader::{VertexShader, FragmentShader, Vertex, GenericShader};
use super::descriptor::DescriptorSetLayout;
use super::command_buffer::CommandBuffer;

use super::errors::{Error, Result};

pub struct Presenter {
    device: Rc<InnerDevice>,
    swapchain: Option<Swapchain>,

    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    last_frame: Instant,
    last_frame_duration: Duration,
    current_swapchain_sync: usize,
    desired_fps: u32,
    current_frame: FrameId,

    present_queue: Rc<Queue>,
}

impl Presenter {
    pub fn get_swapchain_image_formats(device: &Device) -> Result<(vk::Format, vk::Format)> {
        let swapchain_support = super::utils::query_swapchain_support(
            device.inner.physical_device,
            &device.inner.surface_loader,
            device.inner.surface,
        )?;
        let surface_format = choose_swapchain_format(&swapchain_support.formats);
        let depth_format = super::utils::find_depth_format(
            &device.inner.instance,
            device.inner.physical_device,
        )?;
        Ok((surface_format.format, depth_format))
    }

    pub fn new(
        device: &Device,
        render_pass: &RenderPass,
        desired_fps: u32,
    ) -> Result<Self> {
        let swapchain = Swapchain::new(device.inner.clone(), render_pass)?;

        let mut image_available_semaphores = vec![];
        let mut render_finished_semaphores = vec![];

        let semaphore_create_info = vk::SemaphoreCreateInfo{
            s_type: vk::StructureType::SEMAPHORE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::SemaphoreCreateFlags::empty(),
        };

        let num_swapchain_images = swapchain.get_num_images();
        dbg!(num_swapchain_images);

        for _ in 0..num_swapchain_images {
            unsafe {
                let image_available_semaphore = Error::wrap_result(
                    device.inner.device
                        .create_semaphore(&semaphore_create_info, None),
                    "Failed to create image-available semaphore",
                )?;
                let render_finished_semaphore = Error::wrap_result(
                    device.inner.device
                        .create_semaphore(&semaphore_create_info, None),
                    "Failed to create render-finished semaphore",
                )?;

                image_available_semaphores
                    .push(image_available_semaphore);
                render_finished_semaphores
                    .push(render_finished_semaphore);
            }
        }

        Ok(Self{
            device: device.inner.clone(),
            swapchain: Some(swapchain),

            image_available_semaphores,
            render_finished_semaphores,
            current_swapchain_sync: 0,
            last_frame: Instant::now(),
            last_frame_duration: Duration::new(0, 0),
            desired_fps,
            current_frame: FrameId::initial(),

            present_queue: device.inner.get_default_present_queue(),
        })
    }

    pub fn get_dimensions(&self) -> (usize, usize) {
        self.swapchain.as_ref().unwrap().get_dimensions()
    }

    pub fn get_render_extent(&self) -> vk::Extent2D {
        self.swapchain.as_ref().unwrap().swapchain_extent
    }

    pub (crate) fn get_framebuffer(&self) -> vk::Framebuffer {
        self.swapchain.as_ref().unwrap().framebuffer.framebuffer
    }

    pub (crate) fn get_swapchain_image_view(&self, idx: usize) -> vk::ImageView {
        self.swapchain.as_ref().unwrap().frames[idx].imageview.view
    }

    #[allow(unused)]
    pub fn set_desired_fps(&mut self, desired_fps: u32) {
        self.desired_fps = desired_fps;
    }

    pub fn get_current_fps(&self) -> u32 {
        let ns = self.last_frame_duration.as_nanos();
        if ns == 0 {
            0
        } else {
            let fps = 1_000_000_000_f64 / (ns as f64);
            fps as u32
        }
    }

    pub fn get_current_frame(&self) -> FrameId {
        self.current_frame
    }

    pub fn wait_for_next_frame(&self) -> Result<Duration> {
        let millis_since_last_frame = self.last_frame.elapsed().as_millis() as i64;
        let millis_until_next_frame = (
            ((1_f32 / self.desired_fps as f32) * 1000_f32) as i64
        ) - millis_since_last_frame;
        if millis_until_next_frame > 2 {
            //println!("Sleeping {}ms", millis_until_next_frame);
            std::thread::sleep(Duration::from_millis(millis_until_next_frame as u64));
        }

        Ok(self.last_frame.elapsed())
    }

    pub fn acquire_next_image<F>(&mut self, render_pass: &RenderPass, viewport_update: &mut F) -> Result<u32>
    where
        F: FnMut(usize, usize) -> Result<()>
    {
        let (image_index, _is_sub_optimal) = {
            let result = self.device
                .acquire_next_image(
                    self.swapchain.as_ref().unwrap().swapchain,
                    std::u64::MAX,
                    self.image_available_semaphores[self.current_swapchain_sync],
                    vk::Fence::null(),
                );
            match result {
                Ok(res) => res,
                Err(vk_result) => match vk_result {
                    vk::Result::ERROR_OUT_OF_DATE_KHR => {
                        self.fit_to_window(render_pass)?;
                        let (width, height) = self.get_dimensions();
                        viewport_update(width, height)?;
                        // TODO: I hope I don't regret doing this.
                        return self.acquire_next_image(render_pass, viewport_update);
                    },
                    e => return Err(Error::wrap(e, "Failed to acquire swap chain image")),
                },
            }
        };

        Ok(image_index)
    }

    // This is allegedly not even close to complete.  I disagree.
    // The Vulkan tutorial says we need to re-create the following:
    // - swapchain
    // - image views
    // - render pass
    // - graphics pipeline
    // - framebuffers
    // - command buffers
    //
    // Of those, this re-creates the swapchain, image views, and framebuffers.
    // Discussion on Reddit indicates that the render pass only needs to be
    // re-created if the new swapchain images have a different format.
    // WHY THE FUCK WOULD THAT EVER HAPPEN?  Changing the color depth?
    // What is this, 1998?  I had to look up how to do that on a modern OS.
    // I'm not going to all that trouble to support people doing weird shit.
    // Fuck that.  It looks like I can't avoid it for the pipeline, so I'll
    // have to figure out how to signal the engine to do that.
    // Same deal with the command buffers.
    pub fn fit_to_window(&mut self, render_pass: &RenderPass) -> Result<()> {
        unsafe {
            Error::wrap_result(
                self.device.device
                    .device_wait_idle(),
                "Failed to wait for device idle",
            )?;
        }

        self.swapchain = None;
        self.swapchain = Some(Swapchain::new(
            self.device.clone(),
            render_pass,
        )?);

        Ok(())
    }

    pub fn submit_command_buffer(
        &self,
        command_buffer: &CommandBuffer,
        wait_stage: vk::PipelineStageFlags,
    ) -> Result<()> {
        let idx = self.current_swapchain_sync;
        command_buffer.submit_synced(
            &[(wait_stage, self.image_available_semaphores[idx])],
            &[self.render_finished_semaphores[idx]],
        )
    }

    pub fn present_frame<F>(&mut self, image_index: u32, render_pass: &RenderPass, viewport_update: &mut F) -> Result<()>
    where
        F: FnMut(usize, usize) -> Result<()>
    {
        //println!("Presenting a frame...");
        //let start = std::time::Instant::now();
        let swapchains = [self.swapchain.as_ref().unwrap().swapchain];
        let wait_semaphores = [self.render_finished_semaphores[self.current_swapchain_sync]];
        let present_info = vk::PresentInfoKHR{
            s_type: vk::StructureType::PRESENT_INFO_KHR,
            p_next: ptr::null(),
            wait_semaphore_count: 1,
            p_wait_semaphores: wait_semaphores.as_ptr(),
            swapchain_count: 1,
            p_swapchains: swapchains.as_ptr(),
            p_image_indices: &image_index,
            p_results: ptr::null_mut(),
        };

        let result = self.device.queue_present(Rc::clone(&self.present_queue), &present_info);

        let is_resized = match result {
            Ok(_) => false,
            Err(vk_result) => match vk_result {
                vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::SUBOPTIMAL_KHR => true,
                e => return Err(Error::wrap(e, "Failed to submit swap to presentation queue")),
            },
        };

        if is_resized {
            self.fit_to_window(render_pass)?;
            let (width, height) = self.get_dimensions();
            viewport_update(width, height)?;
        }

        self.current_swapchain_sync = (self.current_swapchain_sync + 1) % self.swapchain.as_ref().unwrap().get_num_images();
        self.last_frame_duration = self.last_frame.elapsed();
        self.last_frame = Instant::now();
        //println!("Presented frame in {}ns", start.elapsed().as_nanos());
        self.current_frame.advance();
        Ok(())
    }
}

impl Drop for Presenter {
    fn drop(&mut self) {
        //println!("Dropping a Presenter...");
        unsafe {
            for sem in self.image_available_semaphores.iter() {
                self.device.device.destroy_semaphore(*sem, None);
            }
            for sem in self.render_finished_semaphores.iter() {
                self.device.device.destroy_semaphore(*sem, None);
            }
        }
    }
}

pub fn choose_swapchain_format(
    available_formats: &Vec<vk::SurfaceFormatKHR>,
) -> vk::SurfaceFormatKHR {
    for &available_format in available_formats.iter() {
        if available_format.format == vk::Format::B8G8R8A8_SRGB
            && available_format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR {
                return available_format.clone();
            }
    }

    available_formats.first().unwrap().clone()
}

pub fn choose_swapchain_present_mode(
    available_present_modes: &Vec<vk::PresentModeKHR>,
) -> vk::PresentModeKHR {
    for &available_present_mode in available_present_modes.iter() {
        if available_present_mode == vk::PresentModeKHR::MAILBOX {
            return available_present_mode;
        }
    }

    vk::PresentModeKHR::FIFO
}

fn choose_swapchain_extent(
    device: Rc<InnerDevice>,
    capabilities: &vk::SurfaceCapabilitiesKHR,
) -> vk::Extent2D {
    if capabilities.current_extent.width != u32::max_value() {
        capabilities.current_extent
    } else {
        use num::clamp;

        let window_size = device.window.inner_size();

        vk::Extent2D{
            width: clamp(
                window_size.width,
                capabilities.min_image_extent.width,
                capabilities.max_image_extent.width,
            ),
            height: clamp(
                window_size.height,
                capabilities.min_image_extent.height,
                capabilities.max_image_extent.height,
            ),
        }
    }
}

struct FrameData {
    _frame_index: u32,
    _image: Image,
    imageview: ImageView,
}

struct Swapchain {
    device: Rc<InnerDevice>,
    swapchain: vk::SwapchainKHR,
    pub (crate) swapchain_extent: vk::Extent2D,
    frames: Vec<FrameData>,
    framebuffer: Framebuffer,
}

impl Swapchain {
    fn new(
        device: Rc<InnerDevice>,
        render_pass: &RenderPass,
    ) -> Result<Self> {
        let swapchain_support = device.query_swapchain_support()?;

        let surface_format = choose_swapchain_format(&swapchain_support.formats);
        let present_mode = choose_swapchain_present_mode(&swapchain_support.present_modes);
        let extent = choose_swapchain_extent(device.clone(), &swapchain_support.capabilities);

        let image_count = swapchain_support.capabilities.min_image_count + 1;
        let image_count = if swapchain_support.capabilities.max_image_count > 0 {
            image_count.min(swapchain_support.capabilities.max_image_count)
        } else {
            image_count
        };
        dbg!(swapchain_support.capabilities.min_image_count);
        dbg!(swapchain_support.capabilities.max_image_count);
        dbg!(image_count);

        let (image_sharing_mode, queue_family_indices) =
            if device.get_default_graphics_queue() != device.get_default_present_queue() {
                (
                    vk::SharingMode::CONCURRENT,
                    vec![
                        device.get_default_graphics_queue().family_idx,
                        device.get_default_present_queue().family_idx,
                    ],
                )
            } else {
                (vk::SharingMode::EXCLUSIVE, vec![])
            };
        let queue_family_index_count = queue_family_indices.len() as u32;

        let swapchain_create_info = vk::SwapchainCreateInfoKHR{
            s_type: vk::StructureType::SWAPCHAIN_CREATE_INFO_KHR,
            p_next: ptr::null(),
            flags: vk::SwapchainCreateFlagsKHR::empty(),
            surface: device.surface,
            min_image_count: image_count,
            image_color_space: surface_format.color_space,
            image_format: surface_format.format,
            image_extent: extent,
            image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            image_sharing_mode,
            p_queue_family_indices: queue_family_indices.as_ptr(),
            queue_family_index_count,
            pre_transform: swapchain_support.capabilities.current_transform,
            composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
            present_mode,
            clipped: vk::TRUE,
            old_swapchain: vk::SwapchainKHR::null(),
            image_array_layers: 1,
        };

        let swapchain = device.create_swapchain(&swapchain_create_info)?;

        let swapchain_images = device.get_swapchain_images(swapchain)?;

        let mut frames = Vec::new();
        let mut frame_index: u32 = 0;
        for swapchain_image in swapchain_images.iter() {
            let image = Image::from_vk_image(
                device.clone(),
                *swapchain_image,
                vk::Extent3D{
                    width: extent.width,
                    height: extent.height,
                    depth: 0,
                },
                surface_format.format,
                vk::ImageType::TYPE_2D,
            );
            let imageview = ImageView::from_image(
                &image,
                vk::ImageAspectFlags::COLOR,
                1,
            )?;

            frames.push(FrameData{
                _frame_index: frame_index,
                _image: image,
                imageview,
            });
            frame_index += 1;
        }

        let framebuffer = Framebuffer::new(
            device.clone(),
            extent.width,
            extent.height,
            &render_pass,
        )?;

        Ok(Self{
            device,
            swapchain,
            swapchain_extent: extent,
            frames,
            framebuffer,
        })
    }

    fn get_num_images(&self) -> usize {
        self.frames.len()
    }

    fn get_dimensions(&self) -> (usize, usize) {
        (self.swapchain_extent.width as usize, self.swapchain_extent.height as usize)
    }

    /*fn replace(&mut self, other: Self) {
        unsafe {
            self.swapchain_loader.destroy_swapchain(self.swapchain, None);

            self.swapchain_loader = other.swapchain_loader;
            self.swapchain = other.swapchain;
            self.swapchain_format = other.swapchain_format;
            self.swapchain_extent = other.swapchain_extent;
            self.frames = other.frames;

            self.color_image = other.color_image;
            self.color_image_view = other.color_image_view;
            self.depth_image = other.depth_image;
            self.depth_image_view = other.depth_image_view;
        }
        std::mem::forget(other);
    }*/
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        self.device.destroy_swapchain(self.swapchain);
    }
}

pub struct PipelineParameters {
    msaa_samples: vk::SampleCountFlags,
    cull_mode: vk::CullModeFlags,
    front_face: vk::FrontFace,
    primitive_restart_enable: vk::Bool32,
    topology: vk::PrimitiveTopology,
    depth_test_enable: vk::Bool32,
    depth_write_enable: vk::Bool32,
    depth_compare_op: vk::CompareOp,
    subpass: SubpassRef,
}

impl PipelineParameters {
    pub fn new(subpass: SubpassRef) -> Self {
        Self{
            msaa_samples: vk::SampleCountFlags::TYPE_1,
            cull_mode: vk::CullModeFlags::BACK,
            front_face: vk::FrontFace::COUNTER_CLOCKWISE,
            primitive_restart_enable: vk::FALSE,
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            depth_test_enable: vk::FALSE,
            depth_write_enable: vk::FALSE,
            depth_compare_op: vk::CompareOp::ALWAYS,
            subpass,
        }
    }

    pub fn with_msaa_samples(mut self, msaa_samples: vk::SampleCountFlags) -> Self {
        self.msaa_samples = msaa_samples;
        self
    }

    pub fn with_cull_mode(mut self, cull_mode: vk::CullModeFlags) -> Self {
        self.cull_mode = cull_mode;
        self
    }

    pub fn with_front_face(mut self, front_face: vk::FrontFace) -> Self {
        self.front_face = front_face;
        self
    }

    #[allow(unused)]
    pub fn with_topology(mut self, topology: vk::PrimitiveTopology) -> Self {
        self.topology = topology;
        self
    }

    pub fn with_depth_compare_op(mut self, depth_compare_op: vk::CompareOp) -> Self {
        self.depth_compare_op = depth_compare_op;
        self
    }

    #[allow(unused)]
    pub fn with_primitive_restart(mut self) -> Self {
        self.primitive_restart_enable = vk::TRUE;
        self
    }

    pub fn with_depth_test(mut self) -> Self {
        self.depth_test_enable = vk::TRUE;
        self
    }

    pub fn with_depth_write(mut self) -> Self {
        self.depth_write_enable = vk::TRUE;
        self
    }
}

pub struct Pipeline<V>
where
    V: Vertex,
{
    device: Rc<InnerDevice>,
    pipeline_layout: vk::PipelineLayout,
    pub (crate) pipeline: RefCell<vk::Pipeline>,
    vert_shader: VertexShader<V>,
    frag_shader: FragmentShader,
    params: PipelineParameters,
}

impl<V: Vertex> super::GraphicsResource for Pipeline<V> {}

impl<V> Pipeline<V>
where
    V: Vertex,
{
    fn from_inner(
        device: Rc<InnerDevice>,
        viewport_width: usize,
        viewport_height: usize,
        render_pass: &RenderPass,
        vert_shader: VertexShader<V>,
        frag_shader: FragmentShader,
        set_layouts: &[&DescriptorSetLayout],
        params: PipelineParameters,
    ) -> Result<Self> {
        let mut vk_set_layouts = vec![];
        for layout in set_layouts.iter() {
            dbg!(&layout.layout);
            vk_set_layouts.push(layout.layout);
        }

        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo{
            s_type: vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineLayoutCreateFlags::empty(),
            set_layout_count: vk_set_layouts.len() as u32,
            p_set_layouts: vk_set_layouts.as_ptr(),
            push_constant_range_count: 0,
            p_push_constant_ranges: ptr::null(),
        };
        dbg!(&pipeline_layout_create_info);

        let pipeline_layout = unsafe {
            Error::wrap_result(
                device.device
                    .create_pipeline_layout(&pipeline_layout_create_info, None),
                "Failed to create pipeline layout",
            )?
        };

        let vert_shader_module = vert_shader.get_shader().shader;
        let frag_shader_module = frag_shader.get_shader().shader;
        let pipeline = {
            let result = Self::create_pipeline(
                device.clone(),
                viewport_width,
                viewport_height,
                render_pass.render_pass,
                pipeline_layout,
                vert_shader_module,
                frag_shader_module,
                &params,
            );
            match result {
                Ok(pipeline) => RefCell::new(pipeline),
                Err(e) => {
                    unsafe {
                        device.device.destroy_pipeline_layout(pipeline_layout, None);
                    }
                    return Err(e.into());
                },
            }
        };

        Ok(Self{
            device: device.clone(),
            pipeline_layout,
            pipeline,
            vert_shader,
            frag_shader,
            params,
        })
    }

    pub fn new(
        device: &Device,
        viewport_width: usize,
        viewport_height: usize,
        render_pass: &RenderPass,
        vert_shader: VertexShader<V>,
        frag_shader: FragmentShader,
        set_layouts: &[&DescriptorSetLayout],
        params: PipelineParameters,
    ) -> Result<Self> {
        Self::from_inner(
            device.inner.clone(),
            viewport_width,
            viewport_height,
            render_pass,
            vert_shader,
            frag_shader,
            set_layouts,
            params,
        )
    }

    fn create_pipeline(
        device: Rc<InnerDevice>,
        viewport_width: usize,
        viewport_height: usize,
        render_pass: vk::RenderPass,
        pipeline_layout: vk::PipelineLayout,
        vert_shader_module: vk::ShaderModule,
        frag_shader_module: vk::ShaderModule,
        params: &PipelineParameters,
    ) -> Result<vk::Pipeline> {
        let main_function_name = CString::new("main").unwrap();

        let shader_stages = [
            vk::PipelineShaderStageCreateInfo{
                s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
                p_next: ptr::null(),
                flags: vk::PipelineShaderStageCreateFlags::empty(),
                module: vert_shader_module,
                p_name: main_function_name.as_ptr(),
                p_specialization_info: ptr::null(),
                stage: vk::ShaderStageFlags::VERTEX,
            },
            vk::PipelineShaderStageCreateInfo{
                s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
                p_next: ptr::null(),
                flags: vk::PipelineShaderStageCreateFlags::empty(),
                module: frag_shader_module,
                p_name: main_function_name.as_ptr(),
                p_specialization_info: ptr::null(),
                stage: vk::ShaderStageFlags::FRAGMENT,
            },
        ];

        let binding_description = V::get_binding_description();
        let attribute_description = V::get_attribute_descriptions();
        let vertex_attribute_description_count = attribute_description.len() as u32;
        let vertex_binding_description_count = binding_description.len() as u32;
        let p_vertex_attribute_descriptions = if vertex_attribute_description_count == 0 {
            ptr::null()
        } else {
            attribute_description.as_ptr()
        };
        let p_vertex_binding_descriptions = if vertex_binding_description_count == 0 {
            ptr::null()
        } else {
            binding_description.as_ptr()
        };

        let vertex_input_state_create_info = vk::PipelineVertexInputStateCreateInfo{
            s_type: vk::StructureType::PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineVertexInputStateCreateFlags::empty(),
            vertex_attribute_description_count,
            p_vertex_attribute_descriptions,
            vertex_binding_description_count,
            p_vertex_binding_descriptions,
        };
        let vertex_input_assembly_state_info = vk::PipelineInputAssemblyStateCreateInfo{
            s_type: vk::StructureType::PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineInputAssemblyStateCreateFlags::empty(),
            primitive_restart_enable: params.primitive_restart_enable,
            topology: params.topology,
        };

        let viewports = [vk::Viewport{
            x: 0.0,
            y: 0.0,
            width: viewport_width as f32,
            height: viewport_height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        }];

        let scissors = [vk::Rect2D{
            offset: vk::Offset2D{ x: 0, y: 0 },
            extent: vk::Extent2D{
                width: viewport_width as u32,
                height: viewport_height as u32,
            },
        }];

        let viewport_state_create_info = vk::PipelineViewportStateCreateInfo{
            s_type: vk::StructureType::PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineViewportStateCreateFlags::empty(),
            scissor_count: scissors.len() as u32,
            p_scissors: scissors.as_ptr(),
            viewport_count: viewports.len() as u32,
            p_viewports: viewports.as_ptr(),
        };

        let rasterization_state_create_info = vk::PipelineRasterizationStateCreateInfo{
            s_type: vk::StructureType::PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineRasterizationStateCreateFlags::empty(),
            depth_clamp_enable: vk::FALSE,
            cull_mode: params.cull_mode,
            front_face: params.front_face,
            line_width: 1.0,
            polygon_mode: vk::PolygonMode::FILL,
            rasterizer_discard_enable: vk::FALSE,
            depth_bias_clamp: 0.0,
            depth_bias_constant_factor: 0.0,
            depth_bias_enable: vk::FALSE,
            depth_bias_slope_factor: 0.0,
        };

        let multisample_state_create_info = vk::PipelineMultisampleStateCreateInfo{
            s_type: vk::StructureType::PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineMultisampleStateCreateFlags::empty(),
            rasterization_samples: params.msaa_samples,
            sample_shading_enable: vk::FALSE,
            min_sample_shading: 0.0,
            p_sample_mask: ptr::null(),
            alpha_to_one_enable: vk::FALSE,
            alpha_to_coverage_enable: vk::FALSE,
        };

        let stencil_state = vk::StencilOpState{
            fail_op: vk::StencilOp::KEEP,
            pass_op: vk::StencilOp::KEEP,
            depth_fail_op: vk::StencilOp::KEEP,
            compare_op: vk::CompareOp::ALWAYS,
            compare_mask: 0,
            write_mask: 0,
            reference: 0,
        };
        let depth_state_create_info = vk::PipelineDepthStencilStateCreateInfo{
            s_type: vk::StructureType::PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineDepthStencilStateCreateFlags::empty(),
            depth_test_enable: params.depth_test_enable,
            depth_write_enable: params.depth_write_enable,
            depth_compare_op: params.depth_compare_op,
            depth_bounds_test_enable: vk::FALSE,
            stencil_test_enable: vk::FALSE,
            front: stencil_state,
            back: stencil_state,
            max_depth_bounds: 1.0,
            min_depth_bounds: 0.0,
        };

        let color_blend_attachment_states = [vk::PipelineColorBlendAttachmentState{
            blend_enable: vk::TRUE,
            color_write_mask: vk::ColorComponentFlags::all(),
            src_color_blend_factor: vk::BlendFactor::SRC_ALPHA,
            dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
            color_blend_op: vk::BlendOp::ADD,
            src_alpha_blend_factor: vk::BlendFactor::ONE_MINUS_DST_ALPHA,
            dst_alpha_blend_factor: vk::BlendFactor::ONE,
            alpha_blend_op: vk::BlendOp::ADD,
        }];
        let color_blend_state = vk::PipelineColorBlendStateCreateInfo{
            s_type: vk::StructureType::PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineColorBlendStateCreateFlags::empty(),
            logic_op_enable: vk::FALSE,
            logic_op: vk::LogicOp::COPY,
            attachment_count: color_blend_attachment_states.len() as u32,
            p_attachments: color_blend_attachment_states.as_ptr(),
            blend_constants: [0.0, 0.0, 0.0, 0.0],
        };

        let pipeline_create_infos = [vk::GraphicsPipelineCreateInfo{
            s_type: vk::StructureType::GRAPHICS_PIPELINE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineCreateFlags::empty(),
            stage_count: shader_stages.len() as u32,
            p_stages: shader_stages.as_ptr(),
            p_vertex_input_state: &vertex_input_state_create_info,
            p_input_assembly_state: &vertex_input_assembly_state_info,
            p_tessellation_state: ptr::null(),
            p_viewport_state: &viewport_state_create_info,
            p_rasterization_state: &rasterization_state_create_info,
            p_multisample_state: &multisample_state_create_info,
            p_depth_stencil_state: &depth_state_create_info,
            p_color_blend_state: &color_blend_state,
            p_dynamic_state: ptr::null(),
            layout: pipeline_layout,
            render_pass: render_pass,
            subpass: params.subpass.into(),
            base_pipeline_handle: vk::Pipeline::null(),
            base_pipeline_index: -1,
        }];

        let pipelines = unsafe {
            match device.device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    &pipeline_create_infos,
                    None,
                ) {
                    Ok(p) => p,
                    Err((_, res)) => return Err(Error::wrap(res, "Pipeline creation failed")),
                }
        };
        Ok(pipelines[0])
    }

    pub fn update_viewport(
        &self,
        viewport_width: usize,
        viewport_height: usize,
        render_pass: &RenderPass,
    ) -> Result<()> {
        let vert_shader_module = self.vert_shader.get_shader().shader;
        let frag_shader_module = self.frag_shader.get_shader().shader;
        let pipeline = Self::create_pipeline(
            self.device.clone(),
            viewport_width,
            viewport_height,
            render_pass.render_pass,
            self.pipeline_layout,
            vert_shader_module,
            frag_shader_module,
            &self.params,
        )?;
        *self.pipeline.borrow_mut() = pipeline;
        Ok(())
    }

    pub fn get_layout(&self) -> vk::PipelineLayout {
        self.pipeline_layout
    }

    pub (crate) fn get_vk(&self) -> vk::Pipeline {
        *self.pipeline.borrow()
    }
}

impl<V> Drop for Pipeline<V>
where
    V: Vertex,
{
    fn drop(&mut self) {
        unsafe {
            self.device.device.destroy_pipeline(*self.pipeline.borrow_mut(), None);
            self.device.device.destroy_pipeline_layout(self.pipeline_layout, None);
        }
    }
}

pub struct AttachmentDescription {
    is_multisampled: bool,
    usage: vk::ImageUsageFlags,
    format: vk::Format,
    aspect: vk::ImageAspectFlags,
    load_op: vk::AttachmentLoadOp,
    store_op: vk::AttachmentStoreOp,
    stencil_load_op: vk::AttachmentLoadOp,
    stencil_store_op: vk::AttachmentStoreOp,
    initial_layout: vk::ImageLayout,
    final_layout: vk::ImageLayout,
}

impl AttachmentDescription {
    pub fn new(
        is_multisampled: bool,
        usage: vk::ImageUsageFlags,
        format: vk::Format,
        aspect: vk::ImageAspectFlags,
        load_op: vk::AttachmentLoadOp,
        store_op: vk::AttachmentStoreOp,
        stencil_load_op: vk::AttachmentLoadOp,
        stencil_store_op: vk::AttachmentStoreOp,
        initial_layout: vk::ImageLayout,
        final_layout: vk::ImageLayout,
    ) -> Self {
        Self{
            is_multisampled,
            usage,
            format,
            aspect,
            load_op,
            store_op,
            stencil_load_op,
            stencil_store_op,
            initial_layout,
            final_layout,
        }
    }

    #[allow(unused)]
    pub fn standard_color_render_target(
        format: vk::Format,
        is_multisampled: bool,
        usage: vk::ImageUsageFlags,
    ) -> Self {
        Self{
            is_multisampled,
            usage,
            format,
            aspect: vk::ImageAspectFlags::COLOR,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        }
    }

    #[allow(unused)]
    pub fn standard_color_intermediate(
        format: vk::Format,
        is_multisampled: bool,
        usage: vk::ImageUsageFlags,
    ) -> Self {
        Self{
            is_multisampled,
            usage,
            format,
            aspect: vk::ImageAspectFlags::COLOR,
            load_op: vk::AttachmentLoadOp::DONT_CARE,
            store_op: vk::AttachmentStoreOp::STORE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        }
    }

    pub fn standard_color_final(
        format: vk::Format,
        should_clear: bool,
        usage: vk::ImageUsageFlags,
    ) -> Self {
        Self{
            is_multisampled: false,
            usage,
            format,
            aspect: vk::ImageAspectFlags::COLOR,
            load_op: if should_clear {
                vk::AttachmentLoadOp::CLEAR
            } else {
                vk::AttachmentLoadOp::DONT_CARE
            },
            store_op: vk::AttachmentStoreOp::STORE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
        }
    }

    pub fn standard_depth(
        format: vk::Format,
        is_multisampled: bool,
        usage: vk::ImageUsageFlags,
    ) -> Self {
        Self{
            is_multisampled,
            usage,
            format,
            aspect: vk::ImageAspectFlags::DEPTH,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        }
    }

    fn as_vk(&self, msaa_samples: vk::SampleCountFlags) -> vk::AttachmentDescription {
        vk::AttachmentDescription{
            flags: vk::AttachmentDescriptionFlags::empty(),
            format: self.format,
            samples: if self.is_multisampled {
                msaa_samples
            } else {
                vk::SampleCountFlags::TYPE_1
            },
            load_op: self.load_op,
            store_op: self.store_op,
            stencil_load_op: self.stencil_load_op,
            stencil_store_op: self.stencil_store_op,
            initial_layout: self.initial_layout,
            final_layout: self.final_layout,
        }
    }
}

pub struct Subpass {
    pipeline_bind_point: vk::PipelineBindPoint,
    input_attachments: Vec<vk::AttachmentReference>,
    color_attachments: Vec<vk::AttachmentReference>,
    depth_attachment: Option<vk::AttachmentReference>,
    resolve_attachments: Vec<vk::AttachmentReference>,
}

struct SubpassData {
    input_attachments: Pin<Vec<vk::AttachmentReference>>,
    color_attachments: Pin<Vec<vk::AttachmentReference>>,
    // TODO: I don't like making this a vec, but the alternative
    //       will likely be too obtuse/confusing/non-borrow-checker-friendly.
    depth_attachment: Pin<Vec<vk::AttachmentReference>>,
    resolve_attachments: Pin<Vec<vk::AttachmentReference>>,
}

impl Subpass {
    pub fn new(pipeline_bind_point: vk::PipelineBindPoint) -> Self {
        Self{
            pipeline_bind_point,
            input_attachments: Vec::new(),
            color_attachments: Vec::new(),
            depth_attachment: None,
            resolve_attachments: Vec::new(),
        }
    }

    pub fn add_input_attachment(&mut self, att: vk::AttachmentReference) {
        self.input_attachments.push(att);
    }

    pub fn add_color_attachment(
        &mut self,
        att: vk::AttachmentReference,
        att_resolve: Option<vk::AttachmentReference>,
    ) {
        // This function is a bit tricky, since we need to have either one resolve
        // attachment per color attachment or none at all.
        if self.color_attachments.len() > self.resolve_attachments.len()
            && att_resolve.is_some() {
            panic!("This subpass already has color attachments without resolve attachments!");
        }
        if self.color_attachments.len() == self.resolve_attachments.len()
            && self.resolve_attachments.len() != 0
            && att_resolve.is_none() {
            panic!("This subpass requires a resolve attachment for every color attachment!");
        }
        match att.layout {
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL |
            vk::ImageLayout::GENERAL => (),
            _ => panic!(
                "Invalid image layout {:?} for a color attachment!",
                att.layout,
            ),
        }
        self.color_attachments.push(att);
        if let Some(att_resolve) = att_resolve {
            match att_resolve.layout {
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL |
                vk::ImageLayout::GENERAL => (),
                _ => panic!(
                    "Invalid image layout {:?} for a resolve attachment!",
                    att_resolve.layout,
                ),
            }
            self.resolve_attachments.push(att_resolve);
        }
    }

    pub fn set_depth_attachment(&mut self, att: vk::AttachmentReference) {
        match att.layout {
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL |
            vk::ImageLayout::GENERAL => (),
            _ => panic!(
                "Invalid image layout {:?} for a depth attachment!",
                att.layout,
            ),
        }
        self.depth_attachment = Some(att);
    }

    fn to_vk(self) -> (SubpassData, vk::SubpassDescription) {
        let (pipeline_bind_point, subpass_data) = match self {
            Self{
                pipeline_bind_point,
                input_attachments,
                color_attachments,
                depth_attachment,
                resolve_attachments,
            } => {
                (
                    pipeline_bind_point,
                    SubpassData{
                        input_attachments: Pin::new(input_attachments),
                        color_attachments: Pin::new(color_attachments),
                        depth_attachment: if let Some(att) = depth_attachment {
                            Pin::new(vec![att])
                        } else {
                            Pin::new(Vec::new())
                        },
                        resolve_attachments: Pin::new(resolve_attachments),
                    },
                )
            },
        };

        let input_attachment_count = subpass_data.input_attachments.len() as u32;
        let p_input_attachments = if subpass_data.input_attachments.len() == 0 {
            ptr::null()
        } else {
            subpass_data.input_attachments.as_ptr()
        };
        let color_attachment_count = subpass_data.color_attachments.len() as u32;
        let p_color_attachments = if subpass_data.color_attachments.len() == 0 {
            ptr::null()
        } else {
            subpass_data.color_attachments.as_ptr()
        };
        let p_resolve_attachments = if subpass_data.resolve_attachments.len() == 0 {
            ptr::null()
        } else {
            subpass_data.resolve_attachments.as_ptr()
        };
        let p_depth_stencil_attachment = if subpass_data.depth_attachment.len() == 0 {
            ptr::null()
        } else {
            subpass_data.depth_attachment.as_ptr()
        };

        (
            subpass_data,
            vk::SubpassDescription{
                flags: vk::SubpassDescriptionFlags::empty(),
                pipeline_bind_point,
                input_attachment_count,
                p_input_attachments,
                color_attachment_count,
                p_color_attachments,
                p_resolve_attachments,
                p_depth_stencil_attachment,
                preserve_attachment_count: 0,
                p_preserve_attachments: ptr::null(),
            },
        )
    }
}

pub struct AttachmentInfoSet {
    infos: Vec<vk::FramebufferAttachmentImageInfo>,
}

impl AttachmentInfoSet {
    pub fn new(infos: Vec<vk::FramebufferAttachmentImageInfo>) -> Self {
        Self{
            infos,
        }
    }

    pub fn check_attachments(&self, set: &AttachmentSet) -> bool {
        if set.attachments.len() != self.infos.len() {
            return false;
        }

        for (i, info) in self.infos.iter().enumerate() {
            let att = &set.attachments[i];
            let mut ok = true;
            att.foreach(|_, tex| {
                let size = tex.get_extent();
                if size.depth != 1 {
                    ok = false;
                    return Ok(());
                }
                if size.width != info.width || size.height != info.height {
                    ok = false;
                    return Ok(());
                }
                Ok(())
            }).unwrap();
            if !ok {
                return false;
            }
        }
        true
    }
}

pub struct AttachmentSet {
    attachments: Vec<PerFrameSet<Rc<Texture>>>,
}

impl AttachmentSet {
    pub fn for_renderpass(
        render_pass: &RenderPass,
        width: usize,
        height: usize,
        msaa_samples: vk::SampleCountFlags,
    ) -> Result<Self> {
        Ok(Self{
            attachments: Self::create_attachment_textures(
                render_pass,
                width,
                height,
                msaa_samples,
            )?,
        })
    }

    pub fn get(&self, frame: FrameId, att_ref: &AttachmentRef) -> Rc<Texture> {
        // We subtract one from the index because the index is based on the first attachment
        // being the swapchain attachment.
        self.attachments[att_ref.idx - 1].get(frame).clone()
    }

    pub (crate) fn get_image_views(&self, frame: FrameId) -> Vec<vk::ImageView> {
        let mut image_views = Vec::new();
        for att in self.attachments.iter() {
            image_views.push(att.get(frame).image_view.view);
        }
        image_views
    }

    fn create_attachment_textures(
        render_pass: &RenderPass,
        width: usize,
        height: usize,
        msaa_samples: vk::SampleCountFlags,
    ) -> Result<Vec<PerFrameSet<Rc<Texture>>>> {
        let mut textures = Vec::new();
        // Skip the first attachment in the list.  By convention, that one is the swapchain image.
        let att_slice = &render_pass.attachments[1..];
        for (idx, att) in att_slice.iter().enumerate() {
            let texture_name = format!("attachment-image-{}", idx);
            textures.push(PerFrameSet::new(
                |_| {
                    Ok(Rc::new(
                        Texture::from_image_builder_internal(
                            render_pass.device.clone(),
                            att.aspect,
                            1,
                            att.initial_layout,
                            ImageBuilder::new2d(&texture_name, width, height)
                                .with_num_samples(msaa_samples)
                                .with_format(att.format)
                                .with_usage(att.usage)
                        )?
                    ))
                }
            )?);
        }
        Ok(textures)
    }

    pub fn resize(
        &mut self,
        render_pass: &RenderPass,
        width: usize,
        height: usize,
        msaa_samples: vk::SampleCountFlags,
    ) -> Result<()> {
        self.attachments = Self::create_attachment_textures(
            render_pass,
            width,
            height,
            msaa_samples,
        )?;
        Ok(())
    }
}

pub struct AttachmentRef {
    idx: usize,
}

impl AttachmentRef {
    pub fn as_vk(&self, layout: vk::ImageLayout) -> vk::AttachmentReference {
        vk::AttachmentReference{
            attachment: self.idx as u32,
            layout,
        }
    }
}

#[derive(Copy, Clone)]
pub struct SubpassRef {
    idx: usize,
}

impl From<SubpassRef> for u32 {
    fn from(subpass_ref: SubpassRef) -> u32 {
        subpass_ref.idx as u32
    }
}

pub struct RenderPassBuilder {
    attachments: Vec<AttachmentDescription>,
    subpasses: Vec<Subpass>,
    deps: Vec<vk::SubpassDependency>,
}

impl RenderPassBuilder {
    pub fn new(swapchain_att: AttachmentDescription) -> Self {
        Self{
            attachments: vec![swapchain_att],
            subpasses: Vec::new(),
            deps: Vec::new(),
        }
    }

    pub fn get_swapchain_attachment(&self) -> AttachmentRef {
        AttachmentRef{
            idx: 0,
        }
    }

    pub fn add_attachment(&mut self, att: AttachmentDescription) -> AttachmentRef {
        let idx = self.attachments.len();
        self.attachments.push(att);
        AttachmentRef{
            idx,
        }
    }

    pub fn add_subpass(&mut self, subpass: Subpass) -> SubpassRef {
        let att_sets = [
            ("input", &subpass.input_attachments),
            ("color", &subpass.color_attachments),
            ("resolve", &subpass.resolve_attachments),
        ];
        for (att_type, att_set) in att_sets.iter() {
            for (i, att_ref) in att_set.iter().enumerate() {
                if att_ref.attachment as usize > self.attachments.len() {
                    panic!(
                        "Invalid {} attachment {}={} (we only have {})",
                        att_type,
                        i,
                        att_ref.attachment,
                        self.attachments.len(),
                    );
                }
            }
        }

        if let Some(att_ref) = &subpass.depth_attachment {
            if att_ref.attachment as usize > self.attachments.len() {
                panic!(
                    "Invalid depth attachment {} (we only have {})",
                    att_ref.attachment,
                    self.attachments.len(),
                );
            }
        }

        let idx = self.subpasses.len();
        self.subpasses.push(subpass);
        SubpassRef{
            idx,
        }
    }

    pub fn add_standard_entry_dep(&mut self) {
        self.add_dep(vk::SubpassDependency{
            src_subpass: vk::SUBPASS_EXTERNAL,
            dst_subpass: 0,
            src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            src_access_mask: vk::AccessFlags::MEMORY_READ,
            dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_READ
                | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            dependency_flags: vk::DependencyFlags::empty(),
        })
    }

    pub fn add_dep(&mut self, dep: vk::SubpassDependency) {
        if dep.src_subpass != vk::SUBPASS_EXTERNAL
            && dep.src_subpass as usize > self.subpasses.len() {
                panic!(
                    "Invalid source subpass {} (we only have {})",
                    dep.src_subpass,
                    self.subpasses.len(),
                );
            }
        if dep.dst_subpass != vk::SUBPASS_EXTERNAL
            && dep.dst_subpass as usize > self.subpasses.len() {
                panic!(
                    "Invalid destination subpass {} (we only have {})",
                    dep.dst_subpass,
                    self.subpasses.len(),
                );
            }
        self.deps.push(dep);
    }
}

pub struct RenderPass {
    device: Rc<InnerDevice>,
    pub (crate) render_pass: vk::RenderPass,
    attachments: Vec<AttachmentDescription>,
}

impl RenderPass {
    pub fn new(
        device: &Device,
        msaa_samples: vk::SampleCountFlags,
        builder: RenderPassBuilder,
    ) -> Result<Self> {
        let (attachments, vk_attachments, subpasses, _subpass_data, deps) = match builder {
            RenderPassBuilder{
                attachments,
                mut subpasses,
                deps,
            } => {
                let mut vk_attachments = Vec::new();
                let mut vk_subpasses = Vec::new();
                let mut _subpass_data = Vec::new();
                for att in attachments.iter() {
                    vk_attachments.push(att.as_vk(msaa_samples));
                }
                for subpass in subpasses.drain(..) {
                    let (subpass_data, vk_subpass) = subpass.to_vk();
                    _subpass_data.push(subpass_data);
                    vk_subpasses.push(vk_subpass);
                }
                (attachments, vk_attachments, vk_subpasses, _subpass_data, deps)
            }
        };

        let render_pass_create_info = vk::RenderPassCreateInfo{
            s_type: vk::StructureType::RENDER_PASS_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::RenderPassCreateFlags::empty(),
            attachment_count: vk_attachments.len() as u32,
            p_attachments: vk_attachments.as_ptr(),
            subpass_count: subpasses.len() as u32,
            p_subpasses: subpasses.as_ptr(),
            dependency_count: deps.len() as u32,
            p_dependencies: deps.as_ptr(),
        };

        unsafe {
            Ok(Self{
                device: device.inner.clone(),
                render_pass: Error::wrap_result(
                    device.inner.device
                        .create_render_pass(&render_pass_create_info, None),
                    "Failed to create render pass",
                )?,
                attachments,
            })
        }
    }

    fn get_attachment_infos(&self, width: u32, height: u32) -> (Vec<Pin<Vec<vk::Format>>>, AttachmentInfoSet) {
        let mut image_infos = vec![];
        let mut formats = vec![];
        for att in self.attachments.iter() {
            let view_formats = Pin::new(vec![att.format]);
            image_infos.push(vk::FramebufferAttachmentImageInfo{
                s_type: vk::StructureType::FRAMEBUFFER_ATTACHMENT_IMAGE_INFO,
                p_next: ptr::null(),
                flags: vk::ImageCreateFlags::empty(),
                usage: att.usage,
                width: width,
                height: height,
                layer_count: 1,
                view_format_count: view_formats.len() as u32,
                p_view_formats: view_formats.as_ptr(),
            });
            formats.push(view_formats);
        }
        (formats, AttachmentInfoSet::new(image_infos))
    }
}

impl Drop for RenderPass {
    fn drop(&mut self) {
        unsafe {
            self.device.device.destroy_render_pass(self.render_pass, None);
        }
    }
}

pub struct Framebuffer {
    device: Rc<InnerDevice>,
    pub (crate) framebuffer: vk::Framebuffer,
}

impl Framebuffer {
    fn new(
        device: Rc<InnerDevice>,
        width: u32,
        height: u32,
        render_pass: &RenderPass,
    ) -> Result<Self> {
        let (_, attachment_image_infos) = render_pass.get_attachment_infos(width, height);
        let attachments_info = vk::FramebufferAttachmentsCreateInfo{
            s_type: vk::StructureType::FRAMEBUFFER_ATTACHMENTS_CREATE_INFO,
            p_next: ptr::null(),
            attachment_image_info_count: attachment_image_infos.infos.len() as u32,
            p_attachment_image_infos: attachment_image_infos.infos.as_ptr(),
        };

        let framebuffer_create_info = vk::FramebufferCreateInfo{
            s_type: vk::StructureType::FRAMEBUFFER_CREATE_INFO,
            p_next: (&attachments_info as *const _) as *const c_void,
            flags: vk::FramebufferCreateFlags::IMAGELESS,
            render_pass: render_pass.render_pass,
            attachment_count: attachments_info.attachment_image_info_count,
            p_attachments: ptr::null(),
            width,
            height,
            layers: 1,
        };

        Ok(Self{
            device: device.clone(),
            framebuffer: unsafe {
                Error::wrap_result(
                    device.device
                        .create_framebuffer(&framebuffer_create_info, None),
                    "Failed to create framebuffer",
                )?
            },
        })
    }
}

impl Drop for Framebuffer {
    fn drop(&mut self) {
        unsafe {
            self.device.device.destroy_framebuffer(self.framebuffer, None);
        }
    }
}
