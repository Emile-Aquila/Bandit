use crate::algorithms::utils::{argmax, mean, RewardHistory};
use rand::Rng;

fn greedy_policy(rew_history: &RewardHistory) -> u32{
    let mut score_vec : Vec<f64> = Vec::new();
    for reward_list in &rew_history.rewards {
        score_vec.push(mean(reward_list))
    }
    argmax(&score_vec) as u32
}

pub fn epsilon_greedy_policy(rew_history: &RewardHistory, eps: f64) -> u32{
    let mut rng = rand::thread_rng();
    if rng.gen::<f64>() < eps{
        rng.gen_range(0..(rew_history.arm_size as u32))
    }else{
        greedy_policy(rew_history)
    }
}