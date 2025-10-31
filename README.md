# FFT模板匹配库

一个高性能的Rust图像模板匹配库，使用FFT（快速傅里叶变换）算法实现快速、准确的模板匹配。

## 特性

- **高性能**：使用FFT算法，处理2560x1440图像仅需约0.6秒
- **高精度**：基于归一化互相关（NCC）算法，准确率可达100%
- **并行处理**：使用rayon库进行多线程并行计算
- **易于使用**：简洁的API设计，支持多种阈值设置

## 性能指标

- **处理速度**：约1.6匹配/秒（2560x1440图像，200x200模板）
- **内存使用**：约128MB（用于FFT计算）
- **准确率**：100%（在测试数据集上）

## 快速开始

### 添加依赖

在你的`Cargo.toml`中添加：

```toml
[dependencies]
image-matching-rs = "0.1.0"
```

### 基本使用

```rust
use image::open;
use image_matching_rs::FFTTemplateMatcher;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 加载图像和模板
    let screen_image = open("screen.png")?.to_luma8();
    let template_image = open("template.png")?.to_luma8();
    
    // 创建匹配器
    let mut matcher = FFTTemplateMatcher::new();
    
    // 准备模板（需要提供目标图像的尺寸）
    matcher.prepare_template(
        &template_image,
        screen_image.width(),
        screen_image.height()
    )?;
    
    // 执行匹配（阈值0.8表示80%相似度）
    let matches = matcher.find_matches(&screen_image, 0.8)?;
    
    // 处理结果
    for (x, y, correlation) in matches {
        println!("找到匹配: 位置({}, {}), 相关系数: {:.3}", x, y, correlation);
    }
    
    Ok(())
}
```

## API文档

### FFTTemplateMatcher

主要的模板匹配器结构。

#### 方法

##### `new() -> Self`

创建一个新的模板匹配器实例。

##### `prepare_template(&mut self, template: &ImageBuffer<Luma<u8>, Vec<u8>>, image_width: u32, image_height: u32) -> Result<(), String>`

准备模板进行匹配。这个方法会预计算FFT数据以提高后续匹配的性能。

**参数：**
- `template`: 模板图像（灰度图像）
- `image_width`: 目标图像的宽度
- `image_height`: 目标图像的高度

**返回：**
- `Ok(())`: 成功
- `Err(String)`: 错误信息

##### `find_matches(&self, image: &ImageBuffer<Luma<u8>, Vec<u8>>, threshold: f32) -> Result<Vec<MatchResult>, String>`

在图像中查找模板匹配。

**参数：**
- `image`: 要搜索的图像（灰度图像）
- `threshold`: 匹配阈值（0.0-1.0，推荐0.7-0.9）

**返回：**
- `Ok(Vec<MatchResult>)`: 匹配结果列表，按相关系数降序排列
- `Err(String)`: 错误信息

##### `template_size(&self) -> Option<(u32, u32)>`

获取当前模板的尺寸。

**返回：**
- `Some((width, height))`: 模板尺寸
- `None`: 未设置模板

### MatchResult

匹配结果类型别名：`(u32, u32, f64)`

- 第一个元素：匹配位置的x坐标
- 第二个元素：匹配位置的y坐标  
- 第三个元素：相关系数（0.0-1.0）

## 阈值选择指南

- **0.5-0.6**: 宽松匹配，可能包含较多误报
- **0.7-0.8**: 平衡的匹配，适合大多数场景
- **0.9-0.95**: 严格匹配，只返回高度相似的结果
- **0.95+**: 极严格匹配，适合精确匹配场景

## 性能优化建议

1. **使用release模式编译**：`cargo build --release`
2. **合理选择阈值**：较高的阈值可以减少后处理时间
3. **预处理模板**：对于重复使用的模板，只需调用一次`prepare_template`
4. **图像尺寸**：较小的图像和模板可以显著提高性能

## 算法原理

本库使用基于FFT的归一化互相关（NCC）算法：

1. **预处理**：将模板转换到频域并计算共轭
2. **图像处理**：将输入图像转换到频域
3. **卷积**：在频域中执行快速卷积
4. **后处理**：转换回空域并计算归一化相关系数
5. **并行化**：使用多线程并行计算所有位置的相关系数

## 依赖项

- `image`: 图像处理
- `rustfft`: FFT计算
- `rayon`: 并行处理
- `num-complex`: 复数运算

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request！