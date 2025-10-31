use image::open;
use image_matching_rs::{ImageMatcher, MatcherMode};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== 图像匹配性能基准测试 (使用新的ImageMatcher API) ===");
    
    // 加载图像和模板
    let screen_image = open("images/screen.png")?;
    let template_image = open("images/template.png")?.to_luma8();
    
    println!("测试配置:");
    println!("  屏幕图像: {}x{}", screen_image.width(), screen_image.height());
    println!("  模板图像: {}x{}", template_image.width(), template_image.height());
    println!("  图像像素数: {}", screen_image.width() * screen_image.height());
    
    // 创建匹配器
    let mut matcher = ImageMatcher::new();
    
    // 测试模板准备时间
    println!("\n=== 模板准备性能 (FFT模式) ===");
    let start = Instant::now();
    matcher.prepare_template(
        &template_image,
        screen_image.width(),
        screen_image.height(),
        MatcherMode::FFT
    )?;
    let prepare_time = start.elapsed();
    println!("模板准备时间: {:.2}ms", prepare_time.as_millis());
    
    // 预热运行
    println!("\n=== 预热运行 ===");
    let _ = matcher.matching(screen_image.clone(), MatcherMode::FFT, 0.8)?;
    println!("预热完成");
    
    // 性能测试
    println!("\n=== 匹配性能测试 ===");
    let test_rounds = 10;
    let mut total_time = 0u128;
    let mut total_matches = 0;
    
    for i in 1..=test_rounds {
        let start = Instant::now();
        let matches = matcher.matching(screen_image.clone(), MatcherMode::FFT, 0.8)?;
        let elapsed = start.elapsed();
        
        total_time += elapsed.as_millis();
        total_matches += matches.len();
        
        println!("第{}轮: {:.2}ms, 找到{}个匹配", i, elapsed.as_millis(), matches.len());
    }
    
    // 计算统计数据
    let avg_time = total_time as f64 / test_rounds as f64;
    let avg_matches = total_matches as f64 / test_rounds as f64;
    let throughput = 1000.0 / avg_time; // 匹配/秒
    let pixels_per_second = (screen_image.width() * screen_image.height()) as f64 * throughput;
    
    println!("\n=== 性能统计 ===");
    println!("测试轮数: {}", test_rounds);
    println!("平均匹配时间: {:.2}ms", avg_time);
    println!("平均匹配数量: {:.1}", avg_matches);
    println!("处理吞吐量: {:.2} 匹配/秒", throughput);
    println!("像素处理速度: {:.2} M像素/秒", pixels_per_second / 1_000_000.0);
    
    // 内存使用估算
    let fft_size = 4096u32; // 从调试信息得知
    let memory_mb = (fft_size * fft_size * 8) as f64 / 1024.0 / 1024.0;
    println!("估算内存使用: {:.1}MB", memory_mb);
    
    // 不同阈值的性能测试
    println!("\n=== 不同阈值性能测试 ===");
    let thresholds = [0.5, 0.7, 0.8, 0.9, 0.95];
    
    for &threshold in &thresholds {
        let start = Instant::now();
        let matches = matcher.matching(screen_image.clone(), MatcherMode::FFT, threshold)?;
        let elapsed = start.elapsed();
        
        println!("阈值{:.2}: {:.2}ms, {}个匹配", threshold, elapsed.as_millis(), matches.len());
    }
    
    // 分段模式性能测试
    println!("\n=== 分段模式性能测试 ===");
    
    // 准备分段模式模板
    let start = Instant::now();
    matcher.prepare_template(
        &template_image,
        screen_image.width(),
        screen_image.height(),
        MatcherMode::Segmented
    )?;
    let segmented_prepare_time = start.elapsed();
    println!("分段模式模板准备时间: {:.2}ms", segmented_prepare_time.as_millis());
    
    // 分段模式预热
    let screen_luma = screen_image.to_luma8();
    let _ = matcher.match_by_segmented(&screen_luma, 0.8)?;
    println!("分段模式预热完成");
    
    // 分段模式性能测试
    let mut segmented_total_time = 0u128;
    let mut segmented_total_matches = 0;
    
    for i in 1..=test_rounds {
        let start = Instant::now();
        let matches = matcher.match_by_segmented(&screen_luma, 0.8)?;
        let elapsed = start.elapsed();
        
        segmented_total_time += elapsed.as_millis();
        segmented_total_matches += matches.len();
        
        println!("分段模式第{}轮: {:.2}ms, 找到{}个匹配", i, elapsed.as_millis(), matches.len());
    }
    
    // 分段模式统计
    let segmented_avg_time = segmented_total_time as f64 / test_rounds as f64;
    let segmented_avg_matches = segmented_total_matches as f64 / test_rounds as f64;
    let segmented_throughput = 1000.0 / segmented_avg_time;
    let segmented_pixels_per_second = (screen_image.width() * screen_image.height()) as f64 * segmented_throughput;
    
    println!("\n=== 分段模式性能统计 ===");
    println!("平均匹配时间: {:.2}ms", segmented_avg_time);
    println!("平均匹配数量: {:.1}", segmented_avg_matches);
    println!("处理吞吐量: {:.2} 匹配/秒", segmented_throughput);
    println!("像素处理速度: {:.2} M像素/秒", segmented_pixels_per_second / 1_000_000.0);
    
    // 模式对比
    println!("\n=== 模式性能对比 ===");
    println!("FFT模式:");
    println!("  模板准备: {:.2}ms", prepare_time.as_millis());
    println!("  平均匹配: {:.2}ms", avg_time);
    println!("  吞吐量: {:.2} 匹配/秒", throughput);
    
    println!("分段模式:");
    println!("  模板准备: {:.2}ms", segmented_prepare_time.as_millis());
    println!("  平均匹配: {:.2}ms", segmented_avg_time);
    println!("  吞吐量: {:.2} 匹配/秒", segmented_throughput);
    
    let speedup = avg_time / segmented_avg_time;
    if speedup > 1.0 {
        println!("分段模式比FFT模式快 {:.2}x", speedup);
    } else {
        println!("FFT模式比分段模式快 {:.2}x", 1.0 / speedup);
    }
    
    // 分段模式不同阈值测试
    println!("\n=== 分段模式不同阈值性能测试 ===");
    let segmented_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9];
    
    for &threshold in &segmented_thresholds {
        let start = Instant::now();
        let matches = matcher.match_by_segmented(&screen_luma, threshold)?;
        let elapsed = start.elapsed();
        
        println!("阈值{:.1}: {:.2}ms, {}个匹配", threshold, elapsed.as_millis(), matches.len());
    }
    
    println!("\n=== 基准测试完成 ===");
    
    Ok(())
}