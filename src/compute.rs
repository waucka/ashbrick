use ash::vk;

use super::{Device, InnerDevice};
use super::descriptor::DescriptorSetLayout;
use super::errors::{Error, Result};
use super::shader::{ComputeShader, GenericShader};

use std::cell::RefCell;
use std::ffi::CString;
use std::rc::Rc;
use std::ptr;

pub struct ComputePipeline
{
    device: Rc<InnerDevice>,
    pipeline_layout: vk::PipelineLayout,
    pub (crate) pipeline: RefCell<vk::Pipeline>,
    _shader: ComputeShader,
}

impl ComputePipeline
{
    fn from_inner(
        device: Rc<InnerDevice>,
        shader: ComputeShader,
        set_layouts: &[&DescriptorSetLayout],
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

        let shader_module = shader.get_shader().shader;
        let pipeline = {
            let result = Self::create_pipeline(
                device.clone(),
                pipeline_layout,
                shader_module,
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
            _shader: shader,
        })
    }

    pub fn new(
        device: &Device,
        shader: ComputeShader,
        set_layouts: &[&DescriptorSetLayout],
    ) -> Result<Self> {
        Self::from_inner(
            device.inner.clone(),
            shader,
            set_layouts,
        )
    }

    fn create_pipeline(
        device: Rc<InnerDevice>,
        pipeline_layout: vk::PipelineLayout,
        shader_module: vk::ShaderModule,
    ) -> Result<vk::Pipeline> {
        let main_function_name = CString::new("main").unwrap();

        let pipeline_create_infos = [vk::ComputePipelineCreateInfo{
            s_type: vk::StructureType::COMPUTE_PIPELINE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineCreateFlags::empty(),
            stage: vk::PipelineShaderStageCreateInfo{
                s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
                p_next: ptr::null(),
                flags: vk::PipelineShaderStageCreateFlags::empty(),
                module: shader_module,
                p_name: main_function_name.as_ptr(),
                p_specialization_info: ptr::null(),
                stage: vk::ShaderStageFlags::COMPUTE,
            },
            layout: pipeline_layout,
            base_pipeline_handle: vk::Pipeline::null(),
            base_pipeline_index: -1,
        }];

        let pipelines = unsafe {
            match device.device
                .create_compute_pipelines(
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

    pub fn get_layout(&self) -> vk::PipelineLayout {
        self.pipeline_layout
    }

    pub (crate) fn get_vk(&self) -> vk::Pipeline {
        *self.pipeline.borrow()
    }
}

impl Drop for ComputePipeline
{
    fn drop(&mut self) {
        unsafe {
            self.device.device.destroy_pipeline(*self.pipeline.borrow_mut(), None);
            self.device.device.destroy_pipeline_layout(self.pipeline_layout, None);
        }
    }
}
