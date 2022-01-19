//! This module is all about shaders.  These shader types include some
//! tricks to help the user avoid using a shader on the wrong type of
//! data.

use ash::vk;

use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;
use std::mem::size_of;
use std::ptr;

use super::{Device, InnerDevice};

use super::errors::{Error, Result};

/// Compiles GLSL to SPIR-V v1.5 with the Vulkan 1.2 profile using shaderc
pub fn compile_glsl(
    source: &str,
    kind: shaderc::ShaderKind,
    file_name: &str,
    entry_point: &str,
    standard_includes: &HashMap<String, String>,
    relative_includes: &HashMap<String, String>,
) -> Result<Vec<u32>> {
    let mut compiler = match shaderc::Compiler::new() {
        Some(compiler) => compiler,
        None => return Err(Error::ShaderCompilationError("Failed to load shaderc".to_string())),
    };
    let options = match shaderc::CompileOptions::new() {
        Some(mut options) => {
            options.set_target_env(
                shaderc::TargetEnv::Vulkan,
                shaderc::EnvVersion::Vulkan1_2 as u32,
            );
            options.set_target_spirv(
                shaderc::SpirvVersion::V1_5,
            );
            options.set_source_language(
                shaderc::SourceLanguage::GLSL,
            );
            options.set_include_callback(|include_source, include_type, _, _| {
                let include_set = match include_type {
                    shaderc::IncludeType::Standard => standard_includes,
                    shaderc::IncludeType::Relative => relative_includes,
                };
                match include_set.get(include_source) {
                    Some(code) => Ok(shaderc::ResolvedInclude{
                        resolved_name: include_source.to_string(),
                        content: code.clone(),
                    }),
                    None => Err(format!("Failed to find included file: {}", include_source)),
                }
            });
            options
        },
        None => {
            return Err(Error::ShaderCompilationError("Failed to load shaderc options".to_string()));
        },
    };
    let spirv_result = compiler.compile_into_spirv(
        source,
        kind,
        file_name,
        entry_point,
        Some(&options),
    );
    let spirv = match spirv_result {
        Ok(spirv) => spirv,
        Err(e) => {
            return Err(Error::ShaderCompilationError(format!("Failed to compile shader: {}", e)));
        },
    };
    Ok(spirv.as_binary().to_vec())
}

pub trait GenericShader {
    fn get_shader(&self) -> &Shader;
}

pub struct Shader {
    device: Rc<InnerDevice>,
    pub (crate) shader: vk::ShaderModule,
}

impl Shader {
    pub fn as_vk(&self) -> vk::ShaderModule {
        self.shader
    }

    fn from_file(device: Rc<InnerDevice>, file_path: &Path) -> Result<Self> {
        let code_bytes = Error::wrap_io(
            std::fs::read(file_path),
            &format!("Failed to read file {}", file_path.display()),
        )?;
        Shader::from_bytes(device, &code_bytes)
    }

    fn from_bytes(device: Rc<InnerDevice>, code_bytes: &[u8]) -> Result<Self> {
        let shader_module_create_info = vk::ShaderModuleCreateInfo{
            s_type: vk::StructureType::SHADER_MODULE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::ShaderModuleCreateFlags::empty(),
            code_size: code_bytes.len(),
            p_code: code_bytes.as_ptr() as *const u32,
        };

        Ok(Self{
            device: device.clone(),
            shader: unsafe {
                Error::wrap_result(
                    device.device
                        .create_shader_module(&shader_module_create_info, None),
                    "Failed to create shader module",
                )?
            },
        })
    }
}

impl Drop for Shader {
    fn drop(&mut self) {
        unsafe {
            self.device.device.destroy_shader_module(self.shader, None);
        }
    }
}

pub trait Vertex {
    fn get_binding_description() -> Vec<vk::VertexInputBindingDescription>;
    fn get_attribute_descriptions() -> Vec<vk::VertexInputAttributeDescription>;
}

pub struct VertexShader<V> where V: Vertex {
    shader: Shader,
    _phantom: std::marker::PhantomData<V>,
}

impl<V> VertexShader<V> where V: Vertex {
    pub fn from_file(device: &Device, file_path: &Path) -> Result<Rc<Self>> {
        Ok(Rc::new(Self{
            shader: Shader::from_file(device.inner.clone(), file_path)?,
            _phantom: std::marker::PhantomData,
        }))
    }

    pub fn from_bytes(device: &Device, code_bytes: &[u8]) -> Result<Rc<Self>> {
        Ok(Rc::new(Self {
            shader: Shader::from_bytes(device.inner.clone(), code_bytes)?,
            _phantom: std::marker::PhantomData,
        }))
    }

    // This always returns true because your code won't compile
    // if the shader isn't compatible.
    pub fn is_compatible(&self, _vertices: &[V]) -> bool {
        true
    }
}

impl<V> GenericShader for VertexShader<V> where V: Vertex {
    fn get_shader(&self) -> &Shader {
        &self.shader
    }
}

pub struct FragmentShader {
    shader: Shader,
}

impl FragmentShader {
    pub fn from_file(device: &Device, file_path: &Path) -> Result<Rc<Self>> {
        Ok(Rc::new(Self{
            shader: Shader::from_file(device.inner.clone(), file_path)?,
        }))
    }

    pub fn from_bytes(device: &Device, code_bytes: &[u8]) -> Result<Rc<Self>> {
        Ok(Rc::new(Self {
            shader: Shader::from_bytes(device.inner.clone(), code_bytes)?
        }))
    }
}

impl GenericShader for FragmentShader {
    fn get_shader(&self) -> &Shader {
        &self.shader
    }
}

pub struct ComputeShader {
    shader: Shader,
}

impl ComputeShader {
    pub fn from_file(device: &Device, file_path: &Path) -> Result<Rc<Self>> {
        Ok(Rc::new(Self{
            shader: Shader::from_file(device.inner.clone(), file_path)?,
        }))
    }

    pub fn from_bytes(device: &Device, code_bytes: &[u8]) -> Result<Rc<Self>> {
        Ok(Rc::new(Self {
            shader: Shader::from_bytes(device.inner.clone(), code_bytes)?
        }))
    }
}

impl GenericShader for ComputeShader {
    fn get_shader(&self) -> &Shader {
        &self.shader
    }
}

pub (crate) struct SpecializationConstants {
    map: Vec<vk::SpecializationMapEntry>,
    constants: Vec<u32>,
}

impl SpecializationConstants {
    pub (crate) fn new() -> Self {
        Self {
            map: Vec::new(),
            constants: Vec::new(),
        }
    }

    pub (crate) fn add_u32(&mut self, constant_id: u32, value: u32) {
        let idx = self.constants.len();
        self.constants.push(value);
        self.map.push(vk::SpecializationMapEntry{
            constant_id,
            offset: (idx * size_of::<u32>()) as u32,
            size: size_of::<u32>(),
        });
    }

    pub (crate) fn add_i32(&mut self, constant_id: u32, value: i32) {
        let idx = self.constants.len();
        self.constants.push(u32::from_le_bytes(value.to_le_bytes()));
        self.map.push(vk::SpecializationMapEntry{
            constant_id,
            offset: (idx * size_of::<u32>()) as u32,
            size: size_of::<i32>(),
        });
    }

    pub (crate) fn add_f32(&mut self, constant_id: u32, value: f32) {
        let idx = self.constants.len();
        self.constants.push(value.to_bits());
        self.map.push(vk::SpecializationMapEntry{
            constant_id,
            offset: (idx * size_of::<u32>()) as u32,
            size: size_of::<f32>(),
        });
    }

    pub (crate) fn to_vk(&self) -> vk::SpecializationInfo {
        let map_entry_count = self.map.len() as u32;
        let data_size = self.constants.len();
        vk::SpecializationInfo{
            map_entry_count,
            p_map_entries: self.map.as_ptr() as *const vk::SpecializationMapEntry,
            data_size,
            p_data: self.constants.as_ptr() as *const std::ffi::c_void,
        }
    }
}
