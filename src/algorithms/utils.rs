
pub(crate) fn mean(vec: &Vec<f64>) -> f64{
    match vec.iter().len() {
        // 0 => Some(0 as T),
        0 => 0.0_f64,
        _ => vec.iter().sum::<f64>() / (vec.iter().len() as f64),
    }
}

pub(crate) fn argmax<T: Clone + std::cmp::PartialOrd>(vec: &Vec<T>) -> i32 {
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

pub struct RewardHistory {
    pub arm_size: u32,
    pub rewards: Vec<Vec<f64>>,
}

impl RewardHistory{
    pub fn observe(&mut self, arm_id: u32, reward: f64){
        self.rewards[arm_id as usize].push(reward);
    }
}

pub fn build_reward_history(arm_size: u32) -> RewardHistory{
    RewardHistory{
        arm_size,
        rewards: vec![vec![0.0_f64; 0]; arm_size as usize],
    }
}