use image::GrayImage;

/// 积分图结构（含一行一列零填充）
pub(crate) struct IntegralImage(Vec<Vec<u64>>);

impl IntegralImage {
    /// 计算以 `(x, y)` 为左上角，尺寸为 `width x height` 的区域像素和
    pub fn sum_region(&self, x: usize, y: usize, width: usize, height: usize) -> f64 {
        let x1 = x;
        let y1 = y;

        let x2 = x + width;
        let y2 = y + height;
        (self.0[y2][x2] + self.0[y1][x1] - self.0[y1][x2] - self.0[y2][x1]) as f64
    }
}
/// 普通积分图与平方积分图的组合
pub(crate) struct IntegralImages {
    pub integral: IntegralImage,
    pub squared_integral: IntegralImage,
}

impl IntegralImages {
    fn update_map(integral_map: &mut [Vec<u64>], y: usize, x: usize, value: u64, skip: bool) {
        if skip {
            return;
        }

        // 积分图公式：
        // I(x, y) = P(x, y) + I(x-1, y) + I(x, y-1) - I(x-1, y-1)
        integral_map[y + 1][x + 1] = value
            + integral_map[y][x + 1] // I(x, y-1)
            + integral_map[y + 1][x] // I(x-1, y)
            - integral_map[y][x]; // I(x-1, y-1)
    }

    /// 构造函数，计算积分图和平方积分图
    pub fn new(img: &GrayImage, skip_squared: bool) -> Self {
        let width = img.width() as usize;
        let height = img.height() as usize;

        // 初始化积分图
        let mut integral = vec![vec![0u64; width + 1]; height + 1];
        let mut squared_integral = vec![vec![0u64; width + 1]; height + 1];

        // 遍历图像，在一个循环中同时更新两个积分图
        for y in 0..height {
            for x in 0..width {
                // 获取像素值并计算平方
                let pixel_value = img.get_pixel(x as u32, y as u32)[0] as u64;
                let squared_value = pixel_value * pixel_value;

                // 调用辅助方法更新积分图
                Self::update_map(&mut integral, y, x, pixel_value, false);
                Self::update_map(&mut squared_integral, y, x, squared_value, skip_squared);
            }
        }

        Self {
            integral: IntegralImage(integral),
            squared_integral: IntegralImage(squared_integral),
        }
    }
}
