use image::open;
use image_matching_rs::{ImageMatcher, MatcherMode};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== 图像匹配性能基准测试 (所有模板和目标图组合，阈值0.98) ===");
    
    // 定义所有模板和目标图组合
    let template_sizes = vec![50, 100, 150, 200];
    let screen_sizes = vec![(1024, 768), (1920, 1080), (2560, 1440)];
    let threshold = 0.98;
    
    println!("测试配置:");
    println!("  模板尺寸: {:?}", template_sizes);
    println!("  屏幕尺寸: {:?}", screen_sizes);
    println!("  匹配阈值: {}", threshold);
    println!("  总组合数: {}", template_sizes.len() * screen_sizes.len());
    
    let mut total_combinations = 0;
    let mut successful_combinations = 0;
    let mut total_fft_time = 0u128;
    let mut total_segmented_time = 0u128;
    let mut total_fft_matches = 0;
    let mut total_segmented_matches = 0;
    
    // 遍历所有模板和目标图组合
    for &template_size in &template_sizes {
        println!("\n=== 测试模板 {}x{} ===", template_size, template_size);
        
        // 加载模板图
        let template_path = format!("images/template{}x{}.png", template_size, template_size);
        let template_image = match open(&template_path) {
            Ok(img) => img,
            Err(e) => {
                println!("警告: 无法加载模板 {}: {}", template_path, e);
                continue;
            }
        };
        
        println!("模板图像尺寸: {}x{}", template_image.width(), template_image.height());
        
        for &(screen_w, screen_h) in &screen_sizes {
            println!("\n--- 测试屏幕 {}x{} ---", screen_w, screen_h);
            total_combinations += 1;
            
            // 加载目标图
            let screen_path = format!("images/screen{}x{}.png", screen_w, screen_h);
            let screen_image = match open(&screen_path) {
                Ok(img) => img,
                Err(e) => {
                    println!("警告: 无法加载屏幕图像 {}: {}", screen_path, e);
                    continue;
                }
            };
            
            println!("屏幕图像尺寸: {}x{}", screen_image.width(), screen_image.height());
            
            // FFT模式测试
            println!("FFT模式测试:");
            
            // 创建FFT匹配器
            let start = Instant::now();
            let fft_matcher = ImageMatcher::new_from_image(
                template_image.clone(),
                MatcherMode::FFT { 
                    width: screen_image.width(), 
                    height: screen_image.height() 
                },
                None
            );
            let prepare_time = start.elapsed();
            println!("  匹配器创建时间: {:.2}ms", prepare_time.as_millis());
            
            // 执行FFT匹配
            let start = Instant::now();
            match fft_matcher.matching(screen_image.clone(), threshold, None) {
                Ok(matches) => {
                    let match_time = start.elapsed();
                    total_fft_time += match_time.as_millis();
                    total_fft_matches += matches.len();
                    
                    println!("  匹配时间: {:.2}ms", match_time.as_millis());
                    println!("  找到匹配: {} 个", matches.len());
                    
                    if !matches.is_empty() {
                        println!("  最佳匹配: 位置({}, {}), 相关系数: {:.4}", 
                            matches[0].x, matches[0].y, matches[0].correlation);
                    }
                }
                Err(e) => {
                    println!("  FFT匹配失败: {}", e);
                }
            }
            
            // 分段模式测试
            println!("分段模式测试:");
            
            // 创建分段匹配器
            let start = Instant::now();
            let segmented_matcher = ImageMatcher::new_from_image(
                template_image.clone(),
                MatcherMode::Segmented,
                None
            );
            let prepare_time = start.elapsed();
            println!("  匹配器创建时间: {:.2}ms", prepare_time.as_millis());
            
            // 执行分段匹配
            let start = Instant::now();
            match segmented_matcher.matching(screen_image.clone(), threshold, None) {
                Ok(matches) => {
                    let match_time = start.elapsed();
                    total_segmented_time += match_time.as_millis();
                    total_segmented_matches += matches.len();
                    
                    println!("  匹配时间: {:.2}ms", match_time.as_millis());
                    println!("  找到匹配: {} 个", matches.len());
                    
                    if !matches.is_empty() {
                        println!("  最佳匹配: 位置({}, {}), 相关系数: {:.4}", 
                            matches[0].x, matches[0].y, matches[0].correlation);
                    }
                    
                    successful_combinations += 1;
                }
                Err(e) => {
                    println!("  分段匹配失败: {}", e);
                }
            }
        }
    }
    
    // 总体统计
    println!("\n=== 总体性能统计 ===");
    println!("总组合数: {}", total_combinations);
    println!("成功组合数: {}", successful_combinations);
    
    if successful_combinations > 0 {
        let avg_fft_time = total_fft_time as f64 / successful_combinations as f64;
        let avg_segmented_time = total_segmented_time as f64 / successful_combinations as f64;
        let avg_fft_matches = total_fft_matches as f64 / successful_combinations as f64;
        let avg_segmented_matches = total_segmented_matches as f64 / successful_combinations as f64;
        
        println!("\nFFT模式平均性能:");
        println!("  平均匹配时间: {:.2}ms", avg_fft_time);
        println!("  平均匹配数量: {:.1}", avg_fft_matches);
        println!("  总匹配时间: {:.2}ms", total_fft_time);
        
        println!("\n分段模式平均性能:");
        println!("  平均匹配时间: {:.2}ms", avg_segmented_time);
        println!("  平均匹配数量: {:.1}", avg_segmented_matches);
        println!("  总匹配时间: {:.2}ms", total_segmented_time);
        
        // 性能对比
        if avg_fft_time > 0.0 && avg_segmented_time > 0.0 {
            let speedup = avg_fft_time / avg_segmented_time;
            if speedup > 1.0 {
                println!("\n性能对比: 分段模式比FFT模式快 {:.2}x", speedup);
            } else {
                println!("\n性能对比: FFT模式比分段模式快 {:.2}x", 1.0 / speedup);
            }
        }
        
        // 匹配效果对比
        println!("\n匹配效果对比:");
        println!("  FFT模式总匹配数: {}", total_fft_matches);
        println!("  分段模式总匹配数: {}", total_segmented_matches);
    }
    
    println!("\n=== 基准测试完成 ===");
    
    Ok(())
}
