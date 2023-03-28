use plotters::prelude::*;

pub(crate) fn mean(vec: &Vec<f64>) -> f64{
    match vec.iter().len() {
        // 0 => Some(0 as T),
        0 => 0.0_f64,
        _ => vec.iter().sum::<f64>() / (vec.iter().len() as f64),
    }
}

pub(crate) fn argmax<T: Clone + PartialOrd>(vec: &Vec<T>) -> i32 {
    if vec.is_empty(){
        -1
    }else{
        let mut max_id = 0;
        let mut max_value = vec[0].clone();
        for (i, val) in vec.iter().enumerate(){
            if *val > max_value{
                max_id = i;
                max_value = val.clone();
            }
        }
        return max_id as i32;
    }
}

pub fn max<T:Clone + PartialOrd>(vec: &Vec<T>) -> T{
    let mut max_value: T = vec[0].clone();
    for tmp in vec {
        if *tmp > max_value {
            max_value = tmp.clone();
        }
    }
    max_value
}

pub fn min<T:Clone + PartialOrd>(vec: &Vec<T>) -> T{
    let mut min_value: T = vec[0].clone();
    for tmp in vec {
        if *tmp < min_value {
            min_value = tmp.clone();
        }
    }
    min_value
}

pub struct RewardHistory {
    pub arm_size: u32,
    pub rewards: Vec<Vec<f64>>,
    pub means: Vec<f64>,
}

impl RewardHistory{
    pub fn observe(&mut self, arm_id: u32, reward: f64){
        self.rewards[arm_id as usize].push(reward);
        let size: f64 = self.rewards[arm_id as usize].len() as f64;
        self.means[arm_id as usize] = (self.means[arm_id as usize] * (size-1.0) + reward) / size;
    }
}

pub fn build_reward_history(arm_size: u32) -> RewardHistory{
    RewardHistory{
        arm_size,
        rewards: vec![vec![0.0_f64; 0]; arm_size as usize],
        means: vec![0.0_f64; arm_size as usize],
    }
}


pub fn plot_data(x_data: &Vec<f64>, y_data: &Vec<f64>, file_name: &str, caption: &str){
    // 参考 : https://qiita.com/kanna/items/ea5b15f1b4ce0fee2ab3

    let (width, height) = (1080, 720);
    let path = "./".to_owned()+file_name;
    let root
        = BitMapBackend::new(&path, (width, height)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let (x_min, x_max) = x_data.iter().fold(
        (0.0/0.0, 0.0/0.0),
        |(m,n), v| (v.min(m), v.max(n))
    );

    let (y_min, y_max) = y_data.iter().fold(
        (0.0/0.0, 0.0/0.0),
        |(m,n), v| (v.min(m), v.max(n))
    );
    let font = ("monospace", 20);

    let mut chart = ChartBuilder::on(&root)
        .caption(caption, font.into_font()) // キャプションのフォントやサイズ
        .margin(20)                         // 上下左右全ての余白
        .x_label_area_size(30)              // x軸ラベル部分の余白
        .y_label_area_size(30)              // y軸ラベル部分の余白
        .build_cartesian_2d(x_min..x_max, y_min..y_max).unwrap(); // (x, y)軸の範囲

    chart.configure_mesh().draw().unwrap();  // x軸y軸、グリッド線などを描画

    // 折れ線グラフの定義＆描画
    let line_series = LineSeries::new(
        x_data.iter().zip(y_data.iter()).map(|(x, y)| (*x, *y)),
        &RED
    );
    chart.draw_series(line_series).unwrap();
    chart.configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw().unwrap();
    root.present().unwrap();
}