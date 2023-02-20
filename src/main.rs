use plotters::prelude::LogScalable;
use rand_distr::num_traits::ToPrimitive;
use crate::algorithms::epsilon_greedy::epsilon_greedy_policy;
use crate::algorithms::thompson_sampling::ts_policy;
use crate::algorithms::ucb::ucb_policy;
use crate::algorithms::utils::{build_reward_history, RewardHistory};
use crate::bandit::BanditMachine;
mod bandit;
mod algorithms;


fn main(){
    // params
    let T: u32 = 100;
    let rew_sigma: f64 = 1.0;

    // codes
    let mus: Vec<f64> = vec![0.5, 1.0, 2.0, 3.0];
    let machine = bandit::ProbabilisticBanditMachine{
        mus,
        arm: Box::new(bandit::build_gaussian_reward(rew_sigma)),
    };

    let mut rew_history: RewardHistory = build_reward_history(4);
    let mut rew_sum: f64 = 0.0;

    for t in 0..T {
        // let selected_arm: u32 = epsilon_greedy_policy(&rew_history, 0.1_f64);
        // let selected_arm: u32 = ucb_policy(&rew_history, t);
        let selected_arm: u32 = ts_policy(&rew_history, rew_sigma, 0.0, 2.0);
        let rew = machine.get_reward(selected_arm);
        println!("selected arm is {}, rew is {}", selected_arm, rew);
        rew_history.observe(selected_arm, rew);
        rew_sum += rew;
    }
    println!("total rew is {}, miss prob is {}",
             rew_sum, ((T-(rew_history.rewards[3].len() as u32)).as_f64()/ T.as_f64()));
}