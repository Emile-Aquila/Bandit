use crate::algorithms::utils::{RewardHistory, max};

pub struct DMEDAgent {
    arm_size: u32,
    target_arms: Vec<u32>,
    d_model: Box<dyn Fn(f64, f64) -> f64>,
}

impl DMEDAgent {
    pub fn policy(&mut self, t: f64, rew_history: &RewardHistory) -> u32{
        let mut arm: u32 = 0;
        if !self.target_arms.is_empty() {
            arm = self.target_arms[0].clone();
            self.target_arms.remove(0);
        }else{
            println!("Error! target_arms is empty!");
        }
        if self.target_arms.is_empty() {
            let optimal_mu: f64 = max(&rew_history.means);
            self.target_arms = (0..self.arm_size).into_iter().filter(
                |&x| rew_history.rewards[x as usize].len() as f64
                    * (self.d_model)(rew_history.means[x as usize], optimal_mu) <= t.ln()
            ).collect::<Vec<u32>>();
        }
        arm
    }
}

pub fn build_DMEDAgent(arm_size: u32, d_model: Box<dyn Fn(f64, f64) -> f64>) -> DMEDAgent {
    DMEDAgent{
        arm_size,
        target_arms: (0..arm_size).collect::<Vec<u32>>(),
        d_model: Box::new(d_model)
    }
}
