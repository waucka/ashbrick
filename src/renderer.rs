//! This module has become a bit of a hodgepodge.  The graphics pipeline
//! code should probably be split out into a separate module with the
//! compute pipeline code, and the presenter isn't tremendously related
//! to render passes.

use ash::vk;
use log::{trace, info};

use std::cell::RefCell;
use std::ffi::CString;
use std::time::{Instant, Duration};
use std::rc::Rc;
use std::ptr;
use std::os::raw::c_void;
use std::pin::Pin;

use super::{Device, InnerDevice, Queue, QueueFamilyRef, FrameId, PerFrameSet};
use super::image::{Image, ImageView, ImageBuilder};
use super::texture::Texture;
use super::sync::{Semaphore, Fence};
use super::shader::{VertexShader, FragmentShader, Vertex, GenericShader, SpecializationConstants};
use super::descriptor::DescriptorSetLayout;
use super::command_buffer::CommandBuffer;

use super::errors::{Error, Result};

#[derive(Copy, Clone)]
pub struct SwapchainImageRef {
    pub (crate) idx: u32,
}

impl std::fmt::Display for SwapchainImageRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.idx.fmt(f)
    }
}

pub struct Presenter {
    device: Rc<InnerDevice>,
    swapchain: Option<Swapchain>,

    image_available_semaphores: PerFrameSet<Rc<Semaphore>>,
    render_finished_semaphores: PerFrameSet<Rc<Semaphore>>,
    submissions: Vec<Option<CommandBufferSubmissionSet>>,
    last_frame: Instant,
    last_frame_duration: Duration,
    desired_fps: u32,
    current_frame: FrameId,

    graphics_queue_family: QueueFamilyRef,
    present_queue: Queue,
}

impl Presenter {
    // Make sure that graphics_queue_family is the queue family you
    // will be using to draw to swapchain images.
    pub fn new(
        device: &Device,
        desired_fps: u32,
        graphics_queue_family: QueueFamilyRef,
        present_queue: &Queue,
    ) -> Result<Self> {
        let swapchain = Swapchain::new(
            device.inner.clone(),
            graphics_queue_family,
            present_queue,
        )?;


        let image_available_semaphores = PerFrameSet::new(
            |frame_id| {
                let name = format!("image-available-semaphore-frame-{}", frame_id.idx);
                Ok(Rc::new(Semaphore::new(device, &name)?))
            },
        )?;
        let render_finished_semaphores = PerFrameSet::new(
            |frame_id| {
                let name = format!("render_finished-semaphore-frame-{}", frame_id.idx);
                Ok(Rc::new(Semaphore::new(device, &name)?))
            },
        )?;

        let num_swapchain_images = swapchain.get_num_images();
        trace!("num_swapchain_images={}", num_swapchain_images);

        let submissions = vec![None; num_swapchain_images];

        Ok(Self{
            device: device.inner.clone(),
            swapchain: Some(swapchain),

            image_available_semaphores,
            render_finished_semaphores,
            submissions,
            last_frame: Instant::now(),
            last_frame_duration: Duration::new(0, 0),
            desired_fps,
            current_frame: FrameId::initial(),

            graphics_queue_family,
            present_queue: *present_queue,
        })
    }

    pub fn get_dimensions(&self) -> (usize, usize) {
        self.swapchain.as_ref().unwrap().get_dimensions()
    }

    pub fn get_render_extent(&self) -> vk::Extent2D {
        self.swapchain.as_ref().unwrap().swapchain_extent
    }

    pub (crate) fn get_swapchain_image_view(&self, img_ref: SwapchainImageRef) -> vk::ImageView {
        self.swapchain.as_ref().unwrap().frames[img_ref.idx as usize].imageview.view
    }

    pub fn get_num_swapchain_images(&self) -> usize {
        self.swapchain.as_ref().unwrap().frames.len()
    }

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

    fn wait_for_next_frame(&self) -> Duration {
        let millis_since_last_frame = self.last_frame.elapsed().as_millis() as i64;
        let millis_until_next_frame = (
            ((1_f32 / self.desired_fps as f32) * 1000_f32) as i64
        ) - millis_since_last_frame;
        if millis_until_next_frame > 2 {
            trace!("Sleeping {}ms", millis_until_next_frame);
            std::thread::sleep(Duration::from_millis(millis_until_next_frame as u64));
        }

        self.last_frame.elapsed()
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
    pub fn fit_to_window(&mut self) -> Result<()> {
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
            self.graphics_queue_family,
            &self.present_queue,
        )?);

        Ok(())
    }

    pub fn acquire_frame(&mut self) -> Result<Frame>
    {
        let image_available_semaphore = Rc::clone(self.image_available_semaphores.get(self.current_frame));
        let render_finished_semaphore = Rc::clone(self.render_finished_semaphores.get(self.current_frame));
        let swapchain_image = {
            let result = self.device
                .acquire_next_image(
                    self.swapchain.as_ref().unwrap().swapchain,
                    std::u64::MAX,
                    image_available_semaphore.semaphore,
                    vk::Fence::null(),
                );
            match result {
                Ok((idx, is_sub_optimal)) => if is_sub_optimal {
                    Err(Error::NeedResize)
                } else {
                    Ok(SwapchainImageRef{ idx })
                },
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => Err(Error::NeedResize),
                Err(e) => Err(Error::wrap(e, "Failed to acquire swap chain image")),
            }
        }?;
        let frame_id = self.current_frame;
        self.current_frame.advance();
        Ok(Frame{
            submission_set: None,
            swapchain_image,
            frame_id,
            image_available_semaphore,
            render_finished_semaphore,
            wait_for_next_frame: false,
        })
    }

    pub fn forget_submission(&mut self, swapchain_image: SwapchainImageRef) {
        self.submissions[swapchain_image.idx as usize] = None;
    }

    pub fn wait_for_rendering(&mut self, time: u64) -> Result<()> {
        for i in 0..self.submissions.len() {
            if let Some(submission_set) = &self.submissions[i] {
                for submission in &submission_set.submissions {
                    if let Some(fence) = &submission.fence {
                        trace!("Waiting for frame {} to finish rendering", i);
                        fence.wait(time)?;
                    }
                }
            }
            self.submissions[i] = None;
        }
        Ok(())
    }

    /// Submits command buffers and presents swapchain image.
    /// Returns submission results for each submitted framebuffer.
    pub fn present(&mut self, frame: Frame) -> Result<SubmissionResults> {
        let Frame{
            submission_set,
            swapchain_image,
            frame_id,
            image_available_semaphore: _,
            render_finished_semaphore,
            wait_for_next_frame,
        } = frame;
        let presentation_wait_semaphores = [render_finished_semaphore.semaphore];

        let mut results = Vec::new();
        if let Some(submission_set) = &submission_set {
            // Release old resources for this swapchain image
            self.submissions[swapchain_image.idx as usize] = None;

            for submission in &submission_set.submissions {
                let CommandBufferSubmission{
                    command_buffer,
                    queue,
                    wait,
                    signal,
                    fence,
                } = submission;
                let mut wait_list = Vec::new();
                for wait_spec in wait {
                    wait_list.push((wait_spec.wait_stage, Rc::clone(&wait_spec.semaphore)));
                }
                let mut signal_list = Vec::new();
                for signal_spec in signal {
                    signal_list.push(Rc::clone(signal_spec));
                }
                let fence = match fence {
                    Some(f) => Some(Rc::clone(f)),
                    None => None,
                };
                results.push(command_buffer.submit_synced(
                    queue,
                    &wait_list,
                    &signal_list,
                    fence,
                ));
            }
        }
        self.submissions[swapchain_image.idx as usize] = submission_set;

        trace!("Presenting a frame...");
        let start = std::time::Instant::now();
        let swapchains = [self.swapchain.as_ref().unwrap().swapchain];
        let present_info = vk::PresentInfoKHR{
            s_type: vk::StructureType::PRESENT_INFO_KHR,
            p_next: ptr::null(),
            wait_semaphore_count: 1,
            p_wait_semaphores: presentation_wait_semaphores.as_ptr(),
            swapchain_count: 1,
            p_swapchains: swapchains.as_ptr(),
            p_image_indices: &swapchain_image.idx,
            p_results: ptr::null_mut(),
        };

        if wait_for_next_frame {
            self.wait_for_next_frame();
        }

        let presentation_result = match self.device.queue_present(&self.present_queue, &present_info) {
            Ok(_) => Ok(()),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) |
            Err(vk::Result::SUBOPTIMAL_KHR) => Err(Error::NeedResize),
            Err(e) => Err(Error::wrap(e, "Failed to present")),
        };

        self.last_frame_duration = self.last_frame.elapsed();
        self.last_frame = Instant::now();
        trace!("Presented frame in {}ns", start.elapsed().as_nanos());
        Ok(SubmissionResults{
            results,
            presentation_result,
            swapchain_image,
            frame_id,
        })
    }
}

#[derive(Clone)]
struct CommandBufferWaitSpec {
    wait_stage: vk::PipelineStageFlags,
    semaphore: Rc<Semaphore>,
}

impl CommandBufferWaitSpec {
    fn new(
        wait_stage: vk::PipelineStageFlags,
        semaphore: Rc<Semaphore>,
    ) -> Self {
        Self{
            wait_stage,
            semaphore,
        }
    }
}

#[derive(Clone)]
pub struct CommandBufferSubmission {
    command_buffer: Rc<CommandBuffer>,
    queue: Queue,
    wait: Vec<CommandBufferWaitSpec>,
    signal: Vec<Rc<Semaphore>>,
    fence: Option<Rc<Fence>>,
}

impl CommandBufferSubmission {
    pub fn new(
        command_buffer: Rc<CommandBuffer>,
        queue: Queue,
        fence: Option<Rc<Fence>>,
    ) -> Self {
        Self{
            command_buffer,
            queue,
            wait: Vec::new(),
            signal: Vec::new(),
            fence,
        }
    }

    pub fn add_wait(
        &mut self,
        wait_stage: vk::PipelineStageFlags,
        semaphore: Rc<Semaphore>,
    ) {
        self.wait.push(CommandBufferWaitSpec::new(wait_stage, semaphore));
    }

    pub fn add_signal(
        &mut self,
        semaphore: Rc<Semaphore>,
    ) {
        self.signal.push(semaphore);
    }

    pub fn set_fence(
        &mut self,
        fence: Rc<Fence>,
    ) {
        self.fence = Some(fence);
    }
}

#[derive(Clone)]
pub struct CommandBufferSubmissionSet {
    submissions: Vec<CommandBufferSubmission>,
}

impl CommandBufferSubmissionSet {
    pub fn new(
        submissions: Vec<CommandBufferSubmission>,
    ) -> Self {
        Self{
            submissions,
        }
    }
}

pub struct Frame {
    submission_set: Option<CommandBufferSubmissionSet>,
    swapchain_image: SwapchainImageRef,
    frame_id: FrameId,
    image_available_semaphore: Rc<Semaphore>,
    render_finished_semaphore: Rc<Semaphore>,
    wait_for_next_frame: bool,
}

impl Frame {
    pub fn set_submission_set(&mut self, submission_set: CommandBufferSubmissionSet) {
        self.submission_set = Some(submission_set);
    }

    pub fn get_swapchain_image(&self) -> SwapchainImageRef {
        self.swapchain_image
    }

    pub fn get_frame_id(&self) -> FrameId {
        self.frame_id
    }

    pub fn get_image_available_semaphore(&self) -> Rc<Semaphore> {
        Rc::clone(&self.image_available_semaphore)
    }

    pub fn get_render_finished_semaphore(&self) -> Rc<Semaphore> {
        Rc::clone(&self.render_finished_semaphore)
    }

    pub fn enable_wait_for_next_frame(&mut self) {
        self.wait_for_next_frame = true;
    }
}

pub struct SubmissionResults {
    pub results: Vec<Result<()>>,
    pub presentation_result: Result<()>,
    pub swapchain_image: SwapchainImageRef,
    pub frame_id: FrameId,
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
}

impl Swapchain {
    fn new(
        device: Rc<InnerDevice>,
        graphics_queue_family: QueueFamilyRef,
        present_queue: &Queue,
    ) -> Result<Self> {
        let swapchain_support = device.query_swapchain_support()?;

        let surface_format = super::utils::choose_swapchain_format(&swapchain_support.formats);
        let present_mode = super::utils::choose_swapchain_present_mode(&swapchain_support.present_modes);
        let extent = device.choose_swapchain_extent(&swapchain_support.capabilities);

        let image_count = swapchain_support.capabilities.min_image_count + 1;
        let image_count = if swapchain_support.capabilities.max_image_count > 0 {
            image_count.min(swapchain_support.capabilities.max_image_count)
        } else {
            image_count
        };
        trace!("Minimum swapchain images: {}", swapchain_support.capabilities.min_image_count);
        trace!("Maximum swapchain images: {}", swapchain_support.capabilities.max_image_count);
        trace!("Creating {} swapchain images...", image_count);

        let (image_sharing_mode, queue_family_indices) =
            if graphics_queue_family.idx != present_queue.family_idx {
                (
                    vk::SharingMode::CONCURRENT,
                    vec![
                        graphics_queue_family.idx,
                        present_queue.family_idx,
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
        for (idx, swapchain_image) in swapchain_images.iter().enumerate() {
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
                &format!("swapchain-image-{}", idx),
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

        Ok(Self{
            device,
            swapchain,
            swapchain_extent: extent,
            frames,
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

pub struct GraphicsPipelineParameters {
    msaa_samples: vk::SampleCountFlags,
    cull_mode: vk::CullModeFlags,
    front_face: vk::FrontFace,
    primitive_restart_enable: vk::Bool32,
    topology: vk::PrimitiveTopology,
    depth_test_enable: vk::Bool32,
    depth_write_enable: vk::Bool32,
    depth_compare_op: vk::CompareOp,
    subpass: SubpassRef,
    push_constants: Vec<vk::PushConstantRange>,
    spec_constants_vertex: SpecializationConstants,
    spec_constants_fragment: SpecializationConstants,
}

impl GraphicsPipelineParameters {
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
            push_constants: Vec::new(),
            spec_constants_vertex: SpecializationConstants::new(),
            spec_constants_fragment: SpecializationConstants::new(),
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

    pub fn with_topology(mut self, topology: vk::PrimitiveTopology) -> Self {
        self.topology = topology;
        self
    }

    pub fn with_depth_compare_op(mut self, depth_compare_op: vk::CompareOp) -> Self {
        self.depth_compare_op = depth_compare_op;
        self
    }

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

    pub fn with_push_constant(mut self, push_constant: vk::PushConstantRange) -> Self {
        self.push_constants.push(push_constant);
        self
    }

    pub fn with_vertex_spec_constant_u32(mut self, constant_id: u32, value: u32) -> Self {
        self.spec_constants_vertex.add_u32(constant_id, value);
        self
    }

    pub fn with_vertex_spec_constant_i32(mut self, constant_id: u32, value: i32) -> Self {
        self.spec_constants_vertex.add_i32(constant_id, value);
        self
    }

    pub fn with_vertex_spec_constant_f32(mut self, constant_id: u32, value: f32) -> Self {
        self.spec_constants_vertex.add_f32(constant_id, value);
        self
    }

    pub fn with_fragment_spec_constant_u32(mut self, constant_id: u32, value: u32) -> Self {
        self.spec_constants_fragment.add_u32(constant_id, value);
        self
    }

    pub fn with_fragment_spec_constant_i32(mut self, constant_id: u32, value: i32) -> Self {
        self.spec_constants_fragment.add_i32(constant_id, value);
        self
    }

    pub fn with_fragment_spec_constant_f32(mut self, constant_id: u32, value: f32) -> Self {
        self.spec_constants_fragment.add_f32(constant_id, value);
        self
    }
}

pub struct GraphicsPipeline<V>
where
    V: Vertex,
{
    device: Rc<InnerDevice>,
    pipeline_layout: vk::PipelineLayout,
    pub (crate) pipeline: RefCell<vk::Pipeline>,
    vert_shader: Rc<VertexShader<V>>,
    frag_shader: Rc<FragmentShader>,
    params: GraphicsPipelineParameters,
}

impl<V> GraphicsPipeline<V>
where
    V: Vertex,
{
    fn from_inner(
        device: Rc<InnerDevice>,
        viewport_width: usize,
        viewport_height: usize,
        render_pass: &RenderPass,
        vert_shader: Rc<VertexShader<V>>,
        frag_shader: Rc<FragmentShader>,
        set_layouts: &[Rc<DescriptorSetLayout>],
        params: GraphicsPipelineParameters,
    ) -> Result<Self> {
        let mut vk_set_layouts = vec![];
        for layout in set_layouts.iter() {
            vk_set_layouts.push(layout.layout);
        }

        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo{
            s_type: vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineLayoutCreateFlags::empty(),
            set_layout_count: vk_set_layouts.len() as u32,
            p_set_layouts: vk_set_layouts.as_ptr(),
            push_constant_range_count: params.push_constants.len() as u32,
            p_push_constant_ranges: if params.push_constants.is_empty() {
                ptr::null()
            } else {
                params.push_constants.as_ptr()
            },
        };

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
        vert_shader: Rc<VertexShader<V>>,
        frag_shader: Rc<FragmentShader>,
        set_layouts: &[Rc<DescriptorSetLayout>],
        params: GraphicsPipelineParameters,
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
        params: &GraphicsPipelineParameters,
    ) -> Result<vk::Pipeline> {
        let main_function_name = CString::new("main").unwrap();

        let spec_info_vertex = params.spec_constants_vertex.to_vk();
        let p_specialization_info_vertex = if spec_info_vertex.map_entry_count == 0 {
            ptr::null()
        } else {
            &spec_info_vertex
        };

        let spec_info_fragment = params.spec_constants_fragment.to_vk();
        let p_specialization_info_fragment = if spec_info_fragment.map_entry_count == 0 {
            ptr::null()
        } else {
            &spec_info_fragment
        };

        let shader_stages = [
            vk::PipelineShaderStageCreateInfo{
                s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
                p_next: ptr::null(),
                flags: vk::PipelineShaderStageCreateFlags::empty(),
                module: vert_shader_module,
                p_name: main_function_name.as_ptr(),
                p_specialization_info: p_specialization_info_vertex,
                stage: vk::ShaderStageFlags::VERTEX,
            },
            vk::PipelineShaderStageCreateInfo{
                s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
                p_next: ptr::null(),
                flags: vk::PipelineShaderStageCreateFlags::empty(),
                module: frag_shader_module,
                p_name: main_function_name.as_ptr(),
                p_specialization_info: p_specialization_info_fragment,
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
        // TODO: replace this with a simple pipeline holder object that implements Drop.
        unsafe {
            self.device.device.destroy_pipeline(*self.pipeline.borrow_mut(), None);
        }
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

impl<V> Drop for GraphicsPipeline<V>
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

#[derive(Clone)]
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
    clear_value: vk::ClearValue,
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
        clear_value: vk::ClearValue,
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
            clear_value,
        }
    }

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
            clear_value: vk::ClearValue{
                color: vk::ClearColorValue{
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            },
        }
    }

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
            clear_value: vk::ClearValue{
                color: vk::ClearColorValue{
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            },
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
            clear_value: vk::ClearValue{
                color: vk::ClearColorValue{
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            },
        }
    }

    pub fn standard_depth(
        format: vk::Format,
        is_multisampled: bool,
        usage: vk::ImageUsageFlags,
        initial_layout: Option<vk::ImageLayout>,
        final_layout: Option<vk::ImageLayout>,
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
            initial_layout: match initial_layout {
                Some(layout) => layout,
                None => vk::ImageLayout::UNDEFINED,
            },
            final_layout: match final_layout {
                Some(layout) => layout,
                None => vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            },
            clear_value: vk::ClearValue{
                depth_stencil: vk::ClearDepthStencilValue{
                    depth: 1.0,
                    stencil: 0,
                },
            },
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

#[derive(Clone)]
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
    // This MUST exist, since the FramebufferAttachmentImageInfo objects point into it!
    _formats: Vec<Pin<Vec<vk::Format>>>,
    infos: Vec<vk::FramebufferAttachmentImageInfo>,
}

impl AttachmentInfoSet {
    pub fn new(formats: Vec<Pin<Vec<vk::Format>>>, infos: Vec<vk::FramebufferAttachmentImageInfo>) -> Self {
        Self{
            _formats: formats,
            infos,
        }
    }

    pub fn check_attachments(&self, framebuffer: &Framebuffer) -> bool {
        if framebuffer.attachments.len() != self.infos.len() {
            return false;
        }

        for (i, info) in self.infos.iter().enumerate() {
            let att = &framebuffer.attachments[i];
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

#[derive(Copy, Clone, Debug)]
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

#[derive(Clone)]
pub struct RenderPassBuilder {
    attachment_descriptions: Vec<AttachmentDescription>,
    subpasses: Vec<Subpass>,
    deps: Vec<vk::SubpassDependency>,
}

impl RenderPassBuilder {
    pub fn new(swapchain_att: AttachmentDescription) -> Self {
        Self{
            attachment_descriptions: vec![swapchain_att],
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
        let idx = self.attachment_descriptions.len();
        self.attachment_descriptions.push(att);
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
                if att_ref.attachment as usize > self.attachment_descriptions.len() {
                    panic!(
                        "Invalid {} attachment {}={} (we only have {})",
                        att_type,
                        i,
                        att_ref.attachment,
                        self.attachment_descriptions.len(),
                    );
                }
            }
        }

        if let Some(att_ref) = &subpass.depth_attachment {
            if att_ref.attachment as usize > self.attachment_descriptions.len() {
                panic!(
                    "Invalid depth attachment {} (we only have {})",
                    att_ref.attachment,
                    self.attachment_descriptions.len(),
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
    attachment_descriptions: Vec<AttachmentDescription>,
    msaa_samples: vk::SampleCountFlags,
}

impl RenderPass {
    pub fn new(
        device: &Device,
        msaa_samples: vk::SampleCountFlags,
        builder: RenderPassBuilder,
    ) -> Result<Self> {
        let (attachment_descriptions, vk_attachments, subpasses, _subpass_data, deps) = match builder {
            RenderPassBuilder{
                attachment_descriptions,
                mut subpasses,
                deps,
            } => {
                let mut vk_attachments = Vec::new();
                let mut vk_subpasses = Vec::new();
                let mut _subpass_data = Vec::new();
                for att in attachment_descriptions.iter() {
                    vk_attachments.push(att.as_vk(msaa_samples));
                }
                for subpass in subpasses.drain(..) {
                    let (subpass_data, vk_subpass) = subpass.to_vk();
                    _subpass_data.push(subpass_data);
                    vk_subpasses.push(vk_subpass);
                }
                (attachment_descriptions, vk_attachments, vk_subpasses, _subpass_data, deps)
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
                attachment_descriptions,
                msaa_samples,
            })
        }
    }

    fn get_attachment_infos(&self, width: u32, height: u32) -> AttachmentInfoSet {
        let mut image_infos = vec![];
        let mut formats = vec![];
        for att in self.attachment_descriptions.iter() {
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
        AttachmentInfoSet::new(formats, image_infos)
    }

    pub fn get_clear_value(&self, att_ref: AttachmentRef) -> vk::ClearValue {
        self.attachment_descriptions[att_ref.idx].clear_value
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
    // attachments does not include the swapchain image.
    attachments: Vec<PerFrameSet<Rc<Texture>>>,
}

impl Framebuffer {
    fn create(
        render_pass: &RenderPass,
        width: u32,
        height: u32,
    ) -> Result<vk::Framebuffer> {
        let attachment_image_infos = render_pass.get_attachment_infos(width, height);
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

        unsafe {
            Error::wrap_result(
                render_pass.device.device
                    .create_framebuffer(&framebuffer_create_info, None),
                "Failed to create framebuffer",
            )
        }
    }

    fn for_renderpass(
        render_pass: &RenderPass,
        width: u32,
        height: u32,
        msaa_samples: vk::SampleCountFlags,
        framebuffer_generation: usize,
    ) -> Result<Self> {
        let framebuffer = Self::create(render_pass, width, height)?;

        Ok(Self{
            device: Rc::clone(&render_pass.device),
            framebuffer,
            attachments: Self::create_attachment_textures(
                render_pass,
                width,
                height,
                msaa_samples,
                framebuffer_generation,
            )?,
        })
    }

    pub fn get_attachment(&self, frame: FrameId, att_ref: &AttachmentRef) -> Rc<Texture> {
        // We subtract one from the index because the index is based on the first attachment
        // being the swapchain attachment.
        Rc::clone(&self.attachments[att_ref.idx - 1].get(frame))
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
        width: u32,
        height: u32,
        msaa_samples: vk::SampleCountFlags,
        framebuffer_generation: usize,
    ) -> Result<Vec<PerFrameSet<Rc<Texture>>>> {
        info!("Generating new attachment textures for a {}x{} viewport", width, height);
        let mut textures = Vec::new();
        // Skip the first attachment in the list.  By convention, that one is the swapchain image.
        let att_slice = &render_pass.attachment_descriptions[1..];
        for (idx, att) in att_slice.iter().enumerate() {
            textures.push(PerFrameSet::new(
                |frame_id| {
                    let texture_name = format!("attachment-image-{}-frame-{}-gen-{}", idx, frame_id, framebuffer_generation);
                    let image = Image::new_internal(
                        Rc::clone(&render_pass.device),
                        ImageBuilder::new2d(&texture_name, width as usize, height as usize)
                            .with_num_samples(msaa_samples)
                            .with_format(att.format)
                            .with_usage(att.usage),
                    )?;
                    render_pass.device.set_debug_object_name(
                        &image,
                        &texture_name,
                    )?;
                    let image_view = ImageView::from_image(
                        &image,
                        att.aspect,
                        1,
                    )?;
                    Ok(Rc::new(
                        Texture::from_image(
                            Rc::new(image),
                            Rc::new(image_view),
                            1,
                            &texture_name,
                        )?
                    ))
                }
            )?);
        }
        Ok(textures)
    }
}

impl Drop for Framebuffer {
    fn drop(&mut self) {
        unsafe {
            self.device.device.destroy_framebuffer(self.framebuffer, None);
        }
    }
}

pub struct RenderPassData {
    pub (crate) render_pass: RenderPass,
    pub (crate) framebuffer: Framebuffer,
    framebuffer_generation: usize,
}

impl RenderPassData {
    pub fn new(
        render_pass: RenderPass,
        width: u32,
        height: u32,
    ) -> Result<Self> {
        let framebuffer_generation = 0;
        let framebuffer = Framebuffer::for_renderpass(
            &render_pass,
            width,
            height,
            render_pass.msaa_samples,
            framebuffer_generation,
        )?;
        Ok(Self{
            render_pass,
            framebuffer,
            framebuffer_generation: 0,
        })
    }

    pub fn resize(
        &mut self,
        width: u32,
        height: u32,
    ) -> Result<()> {
        self.framebuffer_generation += 1;
        let new_framebuffer = Framebuffer::for_renderpass(
            &self.render_pass,
            width,
            height,
            self.render_pass.msaa_samples,
            self.framebuffer_generation,
        )?;
        self.framebuffer = new_framebuffer;
        Ok(())
    }

    pub fn get_render_pass(&self) -> &RenderPass {
        &self.render_pass
    }

    pub fn get_framebuffer(&self) -> &Framebuffer {
        &self.framebuffer
    }

    pub (crate) fn get_render_pass_vk(&self) -> vk::RenderPass {
        self.render_pass.render_pass
    }

    pub (crate) fn get_framebuffer_vk(&self) -> vk::Framebuffer {
        self.framebuffer.framebuffer
    }
}
