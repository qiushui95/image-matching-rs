//! Image Matching in Rust
//!
//! 提供基于 FFT 的归一化互相关（NCC）模板匹配实现。
//! 核心算法来源于 J.P. Lewis 的论文：
//! "Fast Normalized Cross-Correlation" (http://scribblethink.org/Work/nvisionInterface/vi95_lewis.pdf)
//!
//! ## 使用方法
//!
//! ```rust,no_run
//! use image_matching_rs::{ImageMatcher, MatcherMode};
//! use image::{ImageBuffer, Luma, DynamicImage};
//!
//! // 创建模板和图像
//! let template = ImageBuffer::<Luma<u8>, Vec<u8>>::new(10, 10);
//! let image = DynamicImage::ImageLuma8(ImageBuffer::<Luma<u8>, Vec<u8>>::new(100, 100));
//!
//! let mut fix = ImageMatcher::new();
//! fix.prepare_template(&template, 10, 10, MatcherMode::FFT).unwrap();
//! let results = fix.matching(image, MatcherMode::FFT, 0.8).unwrap();
//! ```
//!
mod matcher;

mod mode;

pub use mode::MatcherMode;

mod template_data;

// pub use template::TemplateData;

mod fft;

mod segmented;

mod filter;

mod result;
