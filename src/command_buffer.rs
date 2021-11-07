use ash::vk;

use std::cell::RefCell;
use std::rc::Rc;
use std::ptr;
use std::os::raw::c_void;

use super::{Device, InnerDevice, Queue, FrameId, GraphicsResource};
use super::descriptor::DescriptorSet;
use super::renderer::{Presenter, RenderPass, Pipeline, AttachmentSet, SubpassRef, Framebuffer};
use super::buffer::{VertexBuffer, IndexBuffer, UploadSourceBuffer, HasBuffer, Buffer};
use super::image::Image;
use super::shader::Vertex;

use super::errors::{Error, Result};

pub struct SecondaryCommandBuffer {
    buf: RefCell<CommandBuffer>,
}

impl super::GraphicsResource for SecondaryCommandBuffer {}

impl SecondaryCommandBuffer {
    pub fn new(
        device: &Device,
        pool: Rc<CommandPool>,
    ) -> Result<Rc<SecondaryCommandBuffer>> {
        let secondary = Rc::new(
            SecondaryCommandBuffer{
                buf: RefCell::new(CommandBuffer::from_inner_device(
                    Rc::clone(&device.inner),
                    vk::CommandBufferLevel::SECONDARY,
                    pool,
                )?),
            });
        Ok(secondary)
    }

    pub fn reset(&self) -> Result<()> {
        self.buf.borrow_mut().reset()
    }

    // TODO: should this be mut?
    pub fn record<T, R>(
        &self,
        usage_flags: vk::CommandBufferUsageFlags,
        render_pass: &RenderPass,
        subpass: SubpassRef,
        write_fn: T,
    ) -> Result<R>
    where
        T: FnMut(&mut BufferWriter) -> Result<R>
    {
        let mut buf = self.buf.borrow_mut();
        let inheritance_info = vk::CommandBufferInheritanceInfo{
            s_type: vk::StructureType::COMMAND_BUFFER_INHERITANCE_INFO,
            p_next: ptr::null(),
            render_pass: render_pass.render_pass,
            subpass: subpass.into(),
            // TODO: see if I can avoid doing this (might have negative effects on performance)
            framebuffer: vk::Framebuffer::null(),
            occlusion_query_enable: vk::FALSE,
            query_flags: vk::QueryControlFlags::empty(),
            // TODO: make this configurable (e.g. debug on/off)
            pipeline_statistics: vk::QueryPipelineStatisticFlags::empty(),
            
        };
        buf.record_internal(
            usage_flags | vk::CommandBufferUsageFlags::RENDER_PASS_CONTINUE,
            true,
            Some(&inheritance_info),
            write_fn,
        )
    }
}

impl Drop for SecondaryCommandBuffer {
    fn drop(&mut self) {
        // Drop has been implemented solely so that SecondaryCommandBuffers can be recorded as
        // dependencies for CommandBuffers.
    }
}

pub struct CommandPool {
    device: Rc<InnerDevice>,
    queue: Rc<Queue>,
    command_pool: vk::CommandPool,
}

impl CommandPool {
    pub fn new(
        device: &Device,
        queue: Rc<Queue>,
        can_reset: bool,
        transient_buffers: bool,
    ) -> Result<Rc<Self>> {
        Self::from_inner(
            Rc::clone(&device.inner),
            queue,
            can_reset,
            transient_buffers,
        )
    }

    pub (crate) fn from_inner(
        device: Rc<InnerDevice>,
        queue: Rc<Queue>,
        can_reset: bool,
        transient_buffers: bool,
    ) -> Result<Rc<Self>> {
        let command_pool_create_info = vk::CommandPoolCreateInfo{
            s_type: vk::StructureType::COMMAND_POOL_CREATE_INFO,
            p_next: ptr::null(),
            flags: {
                let mut flags = vk::CommandPoolCreateFlags::empty();
                if can_reset {
                    flags |= vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER;
                }
                if transient_buffers {
                    flags |= vk::CommandPoolCreateFlags::TRANSIENT;
                }
                flags
            },
            queue_family_index: queue.family_idx,
        };

        let command_pool = unsafe {
            Error::wrap_result(
                device.device.create_command_pool(&command_pool_create_info, None),
                "Failed to create command pool",
            )?
        };

        Ok(Rc::new(Self{
            device,
            queue,
            command_pool,
        }))
    }
}

impl Drop for CommandPool {
    fn drop(&mut self) {
        unsafe {
            self.device.device.destroy_command_pool(self.command_pool, None);
        }
    }
}

pub struct CommandBuffer {
    device: Rc<InnerDevice>,
    pool: Rc<CommandPool>,
    inflight_fence: vk::Fence,
    level: vk::CommandBufferLevel,
    buf: vk::CommandBuffer,
    // This vector stores references to things that shouldn't be destroyed until
    // the command buffer has been destroyed.
    dependencies: Vec<Rc<dyn GraphicsResource>>,
}

impl CommandBuffer {
    pub fn new(
        device: &Device,
        pool: Rc<CommandPool>,
    ) -> Result<Self> {
        CommandBuffer::from_inner_device(device.inner.clone(), vk::CommandBufferLevel::PRIMARY, pool)
    }

    fn from_inner_device(
        device: Rc<InnerDevice>,
        level: vk::CommandBufferLevel,
        pool: Rc<CommandPool>,
    ) -> Result<Self> {
        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo{
            s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
            p_next: ptr::null(),
            command_buffer_count: 1,
            command_pool: pool.command_pool,
            level,
        };

        let fence_create_info = vk::FenceCreateInfo{
            s_type: vk::StructureType::FENCE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::FenceCreateFlags::SIGNALED,
        };
        let inflight_fence = unsafe {
            Error::wrap_result(
                device.device
                    .create_fence(&fence_create_info, None),
                "Failed to create command buffer fence",
            )?
        };

        let command_buffer = unsafe {
            match device.device
                .allocate_command_buffers(&command_buffer_allocate_info) {
                    Ok(buf) => buf,
                    Err(e) => {
                        device.device.destroy_fence(inflight_fence, None);
                        return Err(Error::wrap(e, "Failed to allocate command buffer"));
                    },
                }
        }[0];

        Ok(Self{
            device,
            pool,
            inflight_fence,
            level,
            buf: command_buffer,
            dependencies: Vec::new(),
        })
    }

    pub fn wait_for_ready(&self) -> Result<()> {
        let wait_fences = [self.inflight_fence];
        unsafe {
            Error::wrap_result(
                self.device.device
                    .wait_for_fences(&wait_fences, true, std::u64::MAX),
                "Failed to wait for fences",
            )?;
        }
        Ok(())
    }
    pub fn record<T, R>(
        &mut self,
        usage_flags: vk::CommandBufferUsageFlags,
        write_fn: T,
    ) -> Result<R>
    where
        T: FnMut(&mut BufferWriter) -> Result<R>
    {
        self.record_internal(usage_flags, false, None, write_fn)
    }

    fn record_internal<T, R>(
        &mut self,
        usage_flags: vk::CommandBufferUsageFlags,
        in_render_pass: bool,
        inheritance_info: Option<&vk::CommandBufferInheritanceInfo>,
        mut write_fn: T,
    ) -> Result<R>
    where
        T: FnMut(&mut BufferWriter) -> Result<R>
    {
        let command_buffer_begin_info = vk::CommandBufferBeginInfo{
            s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
            p_next: ptr::null(),
            p_inheritance_info: match inheritance_info {
                Some(p_info) => p_info,
                None => ptr::null(),
            },
            flags: usage_flags,
        };

        unsafe {
            Error::wrap_result(
                self.device.device
                    .begin_command_buffer(self.buf, &command_buffer_begin_info),
                "Failed to begin recording command buffer",
            )?;
        }

        Ok({
            let mut writer = BufferWriter{
                device: self.device.clone(),
                command_buffer: self.buf,
                in_render_pass,
                dependencies: Vec::new(),
            };
            let result = write_fn(&mut writer)?;
            for dep in writer.dependencies.iter() {
                self.dependencies.push(Rc::clone(dep));
            }
            result
        })
    }

    pub fn submit_synced(
        &self,
        wait: &[(vk::PipelineStageFlags, vk::Semaphore)],
        signal_semaphores: &[vk::Semaphore],
    ) -> Result<()> {
        if self.level == vk::CommandBufferLevel::SECONDARY {
            panic!("Tried to manually submit a secondary command buffer!");
        }

        let mut wait_stages = Vec::new();
        let mut wait_semaphores = Vec::new();
        for (stage, sem) in wait {
            wait_stages.push(*stage);
            wait_semaphores.push(*sem);
        }
        let wait_fences = [self.inflight_fence];

        let submit_infos = [vk::SubmitInfo{
            s_type: vk::StructureType::SUBMIT_INFO,
            p_next: ptr::null(),
            wait_semaphore_count: wait_semaphores.len() as u32,
            p_wait_semaphores: wait_semaphores.as_ptr(),
            p_wait_dst_stage_mask: wait_stages.as_ptr(),
            command_buffer_count: 1,
            p_command_buffers: &self.buf,
            signal_semaphore_count: signal_semaphores.len() as u32,
            p_signal_semaphores: signal_semaphores.as_ptr(),
        }];

        unsafe {
            Error::wrap_result(
                self.device.device
                    .reset_fences(&wait_fences),
                "Failed to reset fences during command buffer submission",
            )?;
            Error::wrap_result(
                self.device.device
                    .queue_submit(
                        self.pool.queue.get(),
                        &submit_infos,
                        self.inflight_fence,
                    ),
                "Failed to submit command buffer",
            )?;
        }
        Ok(())
    }

    #[allow(unused)]
    pub fn reset(&mut self) -> Result<()> {
        self.wait_for_ready()?;
        unsafe {
            Error::wrap_result(
                self.device.device.reset_command_buffer(self.buf, vk::CommandBufferResetFlags::empty()),
                "Failed to reset command buffer",
            )?;
        }
        self.dependencies.clear();
        Ok(())
    }

    pub fn submit_and_wait(
        &self,
    ) -> Result<()> {
        if self.level == vk::CommandBufferLevel::SECONDARY {
            panic!("Tried to manually submit a secondary command buffer!");
        }

        self.submit_synced(
            &[],
            &[],
        )?;
        self.wait_for_ready()?;

        Ok(())
    }

    #[allow(unused)]
    pub fn run_oneshot<T>(
        device: &Device,
        pool: Rc<CommandPool>,
        wait_stage: vk::PipelineStageFlags,
        cmd_fn: T,
    ) -> Result<()>
    where
        T: FnMut(&mut BufferWriter) -> Result<()>
    {
        CommandBuffer::run_oneshot_internal(device.inner.clone(), pool, cmd_fn)
    }

    pub (crate) fn run_oneshot_internal<T>(
        device: Rc<InnerDevice>,
        pool: Rc<CommandPool>,
        cmd_fn: T,
    ) -> Result<()>
    where
        T: FnMut(&mut BufferWriter) -> Result<()>
    {
        let mut cmd_buf = CommandBuffer::from_inner_device(
            device,
            vk::CommandBufferLevel::PRIMARY,
            pool,
        )?;
        cmd_buf.record(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT, cmd_fn)?;
        cmd_buf.submit_and_wait()?;
        Ok(())
    }
}

impl Drop for CommandBuffer {
    fn drop(&mut self) {
        let buffers = [self.buf];
        unsafe {
            let wait_fences = [self.inflight_fence];
            match self.device.device
                .wait_for_fences(&wait_fences, true, std::u64::MAX) {
                    Ok(_) => (),
                    Err(e) => println!("Failed to wait for buffer to be ready: {}", e),
                };
            self.device.device.destroy_fence(self.inflight_fence, None);
            self.device.device.free_command_buffers(self.pool.command_pool, &buffers);
        }
    }
}

pub struct BufferWriter {
    device: Rc<InnerDevice>,
    command_buffer: vk::CommandBuffer,
    in_render_pass: bool,
    dependencies: Vec<Rc<dyn GraphicsResource>>,
}

impl BufferWriter {
    pub fn join_render_pass<T, R>(&mut self, mut write_fn: T) -> Result<R>
    where
        T: FnMut(&mut RenderPassWriter) -> Result<R>
    {
        if !self.in_render_pass {
            panic!("join_render_pass() called on a BufferWriter that is not already in a render pass!");
        }

        let mut writer = RenderPassWriter{
            device: self.device.clone(),
            command_buffer: self.command_buffer,
            auto_end: false,
            allow_subpass_increment: false,
            dependencies: Vec::new(),
        };
        let result = write_fn(&mut writer)?;
        for dep in writer.dependencies.iter() {
            self.dependencies.push(Rc::clone(dep));
        }
        Ok(result)
    }

    pub fn begin_render_pass<T, R>(
        &mut self,
        presenter: &Presenter,
        render_pass: &RenderPass,
        framebuffer: &Framebuffer,
        frame: FrameId,
        clear_values: &[vk::ClearValue],
        attachment_set: &AttachmentSet,
        swapchain_image_index: usize,
        first_subpass_uses_secondaries: bool,
        mut write_fn: T,
    ) -> Result<R>
    where
        T: FnMut(&mut RenderPassWriter) -> Result<R>
    {
        if self.in_render_pass {
            panic!("begin_render_pass() called on a BufferWriter that is already in a render pass!");
        }

        let render_area = vk::Rect2D{
            offset: vk::Offset2D{
                x: 0,
                y: 0,
            },
            extent: presenter.get_render_extent()
        };

        let vk_attachments = {
            let mut vk_attachments = vec![];
            vk_attachments.push(presenter.get_swapchain_image_view(swapchain_image_index));
            vk_attachments.extend(attachment_set.get_image_views(frame));
            vk_attachments
        };

        let attachment_info = vk::RenderPassAttachmentBeginInfo{
            s_type: vk::StructureType::RENDER_PASS_ATTACHMENT_BEGIN_INFO,
            p_next: ptr::null(),
            attachment_count: vk_attachments.len() as u32,
            p_attachments: vk_attachments.as_ptr(),
        };

        let render_pass_begin_info = vk::RenderPassBeginInfo{
            s_type: vk::StructureType::RENDER_PASS_BEGIN_INFO,
            p_next: (&attachment_info as *const _) as *const c_void,
            render_pass: render_pass.render_pass,
            framebuffer: framebuffer.framebuffer,
            render_area,
            clear_value_count: clear_values.len() as u32,
            p_clear_values: clear_values.as_ptr(),
        };

        unsafe {
            self.device.device.cmd_begin_render_pass(
                self.command_buffer,
                &render_pass_begin_info,
                if first_subpass_uses_secondaries {
                    vk::SubpassContents::SECONDARY_COMMAND_BUFFERS
                } else {
                    vk::SubpassContents::INLINE
                },
            );
        }

        let mut writer = RenderPassWriter{
            device: self.device.clone(),
            command_buffer: self.command_buffer,
            auto_end: true,
            allow_subpass_increment: true,
            dependencies: Vec::new(),
        };
        let result = write_fn(&mut writer)?;
        for dep in writer.dependencies.iter() {
            self.dependencies.push(Rc::clone(dep));
        }
        Ok(result)
    }

    pub fn pipeline_barrier(
        &mut self,
        src_stage_mask: vk::PipelineStageFlags,
        dst_stage_mask: vk::PipelineStageFlags,
        deps: vk::DependencyFlags,
        memory_barriers: &[vk::MemoryBarrier],
        buffer_memory_barriers: &[vk::BufferMemoryBarrier],
        image_memory_barriers: &[vk::ImageMemoryBarrier],
    ) {
        unsafe {
            self.device.device.cmd_pipeline_barrier(
                self.command_buffer,
                src_stage_mask,
                dst_stage_mask,
                deps,
                memory_barriers,
                buffer_memory_barriers,
                image_memory_barriers,
            );
        }
    }

    pub fn copy_buffer(
        &mut self,
        src_buffer: Rc<Buffer>,
        dst_buffer: Rc<Buffer>,
        copy_regions: &[vk::BufferCopy],
    ) {
        unsafe {
            let src_buf = src_buffer.get_buffer();
            let dst_buf = dst_buffer.get_buffer();
            self.device.device.cmd_copy_buffer(
                self.command_buffer,
                src_buf,
                dst_buf,
                &copy_regions,
            );
        }
        self.dependencies.push(src_buffer);
        self.dependencies.push(dst_buffer);
    }

    pub fn copy_buffer_to_image(
        &mut self,
        src_buffer: Rc<UploadSourceBuffer>,
        image: Rc<Image>,
    ) {
        let buffer_image_regions = [vk::BufferImageCopy{
            image_subresource: vk::ImageSubresourceLayers{
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            image_extent: image.extent,
            buffer_offset: 0,
            buffer_image_height: 0,
            buffer_row_length: 0,
            image_offset: vk::Offset3D{ x: 0, y: 0, z: 0 },
        }];

        unsafe {
            self.device.device.cmd_copy_buffer_to_image(
                self.command_buffer,
                src_buffer.get_buffer(),
                image.img,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &buffer_image_regions,
            );
        }
        self.dependencies.push(src_buffer);
        self.dependencies.push(image);
    }

    pub (crate) unsafe fn copy_buffer_to_image_no_deps(
        &mut self,
        src_buffer: &UploadSourceBuffer,
        image: &Image,
    ) {
        let buffer_image_regions = [vk::BufferImageCopy{
            image_subresource: vk::ImageSubresourceLayers{
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            image_extent: image.extent,
            buffer_offset: 0,
            buffer_image_height: 0,
            buffer_row_length: 0,
            image_offset: vk::Offset3D{ x: 0, y: 0, z: 0 },
        }];

        self.device.device.cmd_copy_buffer_to_image(
            self.command_buffer,
            src_buffer.get_buffer(),
            image.img,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &buffer_image_regions,
        );
    }

    #[allow(unused)]
    pub fn blit_image(
        &mut self,
        img_src: Rc<Image>,
        layout_src: vk::ImageLayout,
        img_dst: Rc<Image>,
        layout_dst: vk::ImageLayout,
        regions: &[vk::ImageBlit],
        filter: vk::Filter,
    ) {
        unsafe {
            self.device.device.cmd_blit_image(
                self.command_buffer,
                img_src.img,
                layout_src,
                img_dst.img,
                layout_dst,
                regions,
                filter,
            );
        }
        self.dependencies.push(img_src);
        self.dependencies.push(img_dst);
    }

    // Dependencies are not recorded.  Use this only in cases where you have no choice!
    pub (crate) unsafe fn blit_image_no_deps(
        &mut self,
        img_src: &Image,
        layout_src: vk::ImageLayout,
        img_dst: &Image,
        layout_dst: vk::ImageLayout,
        regions: &[vk::ImageBlit],
        filter: vk::Filter,
    ) {
        self.device.device.cmd_blit_image(
            self.command_buffer,
            img_src.img,
            layout_src,
            img_dst.img,
            layout_dst,
            regions,
            filter,
        );
    }
}

impl Drop for BufferWriter {
    fn drop(&mut self) {
        unsafe {
            if let Err(e) = self.device.device.end_command_buffer(self.command_buffer) {
                println!("Failed to end command buffer: {:?}", e);
            }
        }
    }
}

pub struct RenderPassWriter {
    device: Rc<InnerDevice>,
    command_buffer: vk::CommandBuffer,
    auto_end: bool,
    allow_subpass_increment: bool,
    dependencies: Vec<Rc<dyn GraphicsResource>>,
}

impl RenderPassWriter {
    pub fn bind_pipeline<V: Vertex + 'static>(
        &mut self,
        pipeline: Rc<Pipeline<V>>,
    ) {
        unsafe {
            self.device.device.cmd_bind_pipeline(
                self.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.get_vk(),
            );
        }
        self.dependencies.push(pipeline);
    }

    pub fn bind_descriptor_sets(
        &mut self,
        pipeline_layout: vk::PipelineLayout,
        descriptor_sets: &[Rc<DescriptorSet>],
    ) {
        let mut vk_sets = Vec::new();
        for set in descriptor_sets {
            vk_sets.push(set.inner);
            // This turbofish horseshit is needed because otherwise, the compiler will
            // infer the type parameter to be `dyn Drop`, and Rc::clone() will barf because
            // its parameter is Rc<DescriptorSet> rather than Rc<dyn Drop>.
            self.dependencies.push(Rc::<DescriptorSet>::clone(set));
        }
        unsafe {
            self.device.device.cmd_bind_descriptor_sets(
                self.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline_layout,
                0,
                &vk_sets,
                &[],
            );
        }
    }

    pub fn draw<T: 'static>(
        &mut self,
        vertex_buffer: Rc<VertexBuffer<T>>,
    ) {
        let vertex_buffers = [vertex_buffer.get_buffer()];
        let offsets = [0_u64];

        unsafe {
            self.device.device.cmd_bind_vertex_buffers(
                self.command_buffer,
                0,
                &vertex_buffers,
                &offsets,
            );
            self.device.device.cmd_draw(
                self.command_buffer,
                vertex_buffer.len() as u32,
                1, 0, 0,
            );
        }
        self.dependencies.push(vertex_buffer);
    }

    pub fn draw_no_vbo(
        &self,
        num_vertices: usize,
        num_instances: usize,
    ) {
        unsafe {
            self.device.device.cmd_draw(
                self.command_buffer,
                num_vertices as u32,
                num_instances as u32,
                0, 0,
            );
        }
    }

    pub fn draw_indexed<T: 'static>(
        &mut self,
        vertex_buffer: Rc<VertexBuffer<T>>,
        index_buffer: Rc<IndexBuffer>,
    ) {
        let vertex_buffers = [vertex_buffer.get_buffer()];
        let offsets = [0_u64];

        unsafe {
            self.device.device.cmd_bind_vertex_buffers(
                self.command_buffer,
                0,
                &vertex_buffers,
                &offsets,
            );
            self.device.device.cmd_bind_index_buffer(
                self.command_buffer,
                index_buffer.get_buffer(),
                0,
                vk::IndexType::UINT32,
            );
            self.device.device.cmd_draw_indexed(
                self.command_buffer,
                index_buffer.len() as u32,
                1, 0, 0, 0,
            );
        }
        self.dependencies.push(vertex_buffer);
        self.dependencies.push(index_buffer);
    }

    pub fn next_subpass(&self, uses_secondaries: bool) {
        if !self.allow_subpass_increment {
            panic!("VkCmdNextSubpass is not allowed on a secondary command buffer!");
        }
        unsafe {
            self.device.device.cmd_next_subpass(
                self.command_buffer,
                if uses_secondaries {
                    vk::SubpassContents::SECONDARY_COMMAND_BUFFERS
                } else {
                    vk::SubpassContents::INLINE
                },
            );
        }
    }

    pub fn execute_commands(&mut self, secondaries: &[Rc<SecondaryCommandBuffer>]) {
        let mut vk_secondaries = Vec::new();
        for secondary in secondaries {
            vk_secondaries.push(secondary.buf.borrow().buf);
            self.dependencies.push(Rc::<SecondaryCommandBuffer>::clone(secondary));
        }
        unsafe {
            self.device.device.cmd_execute_commands(
                self.command_buffer,
                &vk_secondaries,
            );
        }
    }
}

impl Drop for RenderPassWriter {
    fn drop(&mut self) {
        if self.auto_end {
            unsafe {
                self.device.device.cmd_end_render_pass(self.command_buffer);
            }
        }
    }
}
