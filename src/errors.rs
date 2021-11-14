use std::fmt::{Debug, Display, Formatter};
use ash::vk;

#[derive(Debug)]
pub enum Error {
    Generic(String),
    External(Box<dyn std::error::Error + 'static>, String),
    InternalError(String),
    VulkanError{
        vk_result: vk::Result,
        msg: String,
    },
    WindowingError(String),
    NoFormatsAvailable{
        tiling: vk::ImageTiling,
        features: vk::FormatFeatureFlags,
    },
    NoSuitableQueues,
    NoSuitableDevices,
    AllocationError(gpu_allocator::AllocationError),
    UnmappableBuffer,
    InvalidTransition{
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
    },
    InvalidTextureSize{
        data_len: usize,
        required_multiple: usize,
    },
    Io{
        err: std::io::Error,
        msg: String,
    },
    ZeroSizeBuffer,
    InvalidBufferCopy(u64, u64),
    InvalidUniformWrite(usize, usize),
    NeedResize,
    SrgbNotAvailable,
}

impl Error {
    pub fn external<E: std::error::Error + 'static>(err: E, msg: &str) -> Self {
        Error::External(Box::new(err), String::from(msg))
    }

    pub fn internal(msg: &str) -> Self {
        Error::InternalError(String::from(msg))
    }

    pub fn windowing(msg: &str) -> Self {
        Error::WindowingError(String::from(msg))
    }

    pub fn wrap_external<T, E: std::error::Error + 'static>(res: std::result::Result<T, E>, msg: &str) -> Result<T> {
        match res {
            Ok(v) => Ok(v),
            Err(err) => Err(Error::External(Box::new(err), String::from(msg)))
        }
    }

    pub fn wrap_io<T>(res: std::io::Result<T>, msg: &str) -> Result<T> {
        match res {
            Ok(v) => Ok(v),
            Err(err) => Err(Error::Io{
                err,
                msg: String::from(msg),
            }),
        }
    }

    pub fn wrap(vk_result: ash::vk::Result, msg_str: &str) -> Self {
        let msg = String::from(msg_str);
        Error::VulkanError{
            vk_result,
            msg,
        }
    }

    pub fn wrap_result<T>(vk_result: ash::prelude::VkResult<T>, msg_str: &str) -> Result<T> {
        match vk_result {
            Ok(v) => Ok(v),
            Err(vk_err) => Err(Self::wrap(vk_err, msg_str)),
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        use Error::*;
        use std::ops::Deref;
        match self {
            External(err, _) => Some(err.deref()),
            _ => None,
        }
    }
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use Error::*;
        match self {
            Generic(msg) => write!(f, "{}", msg),
            External(err, msg) => write!(f, "{}: {}", err, msg),
            InternalError(msg) => write!(f, "Internal error: {}", msg),
            VulkanError{
                vk_result,
                msg,
            } => {
                write!(f, "{}: {}", msg, vk_result)
            },
            WindowingError(msg) => write!(f, "Windowing error: {}", msg),
            NoFormatsAvailable{
                tiling,
                features,
            } => {
                write!(f, "No formats found with tiling {:?} and features {:?}", tiling, features)
            },
            NoSuitableQueues => write!(f, "Unable to find any queues supporting both graphics and presentation"),
            NoSuitableDevices => write!(f, "Unable to find any suitable physical devices"),
            AllocationError(e) => write!(f, "Failed to allocate memory: {}", e),
            UnmappableBuffer => write!(f, "Buffer cannot be mapped into the host address space"),
            InvalidTransition{
                old_layout,
                new_layout,
            } => write!(f, "Unsupported layout transition {:?} -> {:?}", old_layout, new_layout),
            InvalidTextureSize{
                data_len,
                required_multiple,
            } => {
                write!(f, "Float texture is {} bytes long; which is not a multiple of {}", data_len, required_multiple)
            },
            Io{
                err,
                msg,
            } => write!(f, "{}: {}", msg, err),
            ZeroSizeBuffer => write!(f, "Buffer size must be greater than zero"),
            InvalidBufferCopy(buf1, buf2) => write!(f, "Tried to copy a {} byte buffer into a {} byte buffer", buf1, buf2),
            InvalidUniformWrite(buf1, buf2) => write!(f, "Tried to write {} bytes to a {}-byte uniform buffer", buf1, buf2),
            NeedResize => write!(f, "Framebuffer needs resize"),
            SrgbNotAvailable => write!(f, "Automatic sRGB conversion is not available for 16 bit per channel formats"),
        }
    }
}
