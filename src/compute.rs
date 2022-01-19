//! This module contains compute-specific code.
//! I don't like it.  The compute pipeline code should probably
//! live with the graphics pipeline code.

use ash::vk;

use super::{Device, InnerDevice};
use super::descriptor::DescriptorSetLayout;
use super::errors::{Error, Result};
use super::shader::{ComputeShader, GenericShader, SpecializationConstants};

use std::cell::RefCell;
use std::ffi::CString;
use std::rc::Rc;
use std::ptr;

pub struct ComputePipelineParameters {
    shader: Rc<ComputeShader>,
    set_layouts: Vec<Rc<DescriptorSetLayout>>,
    push_constants: Vec<vk::PushConstantRange>,
    spec_constants: SpecializationConstants,
}

impl ComputePipelineParameters {
    pub fn new(shader: Rc<ComputeShader>) -> Self {
        Self{
            shader,
            set_layouts: Vec::new(),
            push_constants: Vec::new(),
            spec_constants: SpecializationConstants::new(),
        }
    }

    pub fn with_set_layout(mut self, layout: Rc<DescriptorSetLayout>) -> Self {
        self.set_layouts.push(layout);
        self
    }

    pub fn with_push_constant(mut self, push_constant: vk::PushConstantRange) -> Self {
        self.push_constants.push(push_constant);
        self
    }

    pub fn with_spec_constant_u32(mut self, constant_id: u32, value: u32) -> Self {
        self.spec_constants.add_u32(constant_id, value);
        self
    }

    pub fn with_spec_constant_i32(mut self, constant_id: u32, value: i32) -> Self {
        self.spec_constants.add_i32(constant_id, value);
        self
    }

    pub fn with_spec_constant_f32(mut self, constant_id: u32, value: f32) -> Self {
        self.spec_constants.add_f32(constant_id, value);
        self
    }
}

pub struct ComputePipeline {
    device: Rc<InnerDevice>,
    pipeline_layout: vk::PipelineLayout,
    pub (crate) pipeline: RefCell<vk::Pipeline>,
    _shader: Rc<ComputeShader>,
}

impl ComputePipeline {
    fn from_inner(
        device: Rc<InnerDevice>,
        params: ComputePipelineParameters,
    ) -> Result<Self> {
        let ComputePipelineParameters{
            shader,
            set_layouts,
            push_constants,
            spec_constants,
        } = params;

        let mut vk_set_layouts = vec![];
        for layout in set_layouts.iter() {
            vk_set_layouts.push(layout.layout);
        }

        let spec_info = spec_constants.to_vk();

        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo{
            s_type: vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineLayoutCreateFlags::empty(),
            set_layout_count: vk_set_layouts.len() as u32,
            p_set_layouts: vk_set_layouts.as_ptr(),
            push_constant_range_count: push_constants.len() as u32,
            p_push_constant_ranges: if push_constants.is_empty() {
                ptr::null()
            } else {
                push_constants.as_ptr()
            },
        };

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
                if spec_info.map_entry_count == 0 {
                    ptr::null()
                } else {
                    &spec_info
                }
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
        params: ComputePipelineParameters,
    ) -> Result<Self> {
        Self::from_inner(
            device.inner.clone(),
            params,
        )
    }

    fn create_pipeline(
        device: Rc<InnerDevice>,
        pipeline_layout: vk::PipelineLayout,
        shader_module: vk::ShaderModule,
        p_specialization_info: *const vk::SpecializationInfo,
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
                p_specialization_info,
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
