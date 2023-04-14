use crate::algorithms::epsilon_greedy::epsilon_greedy_policy;
use crate::algorithms::exp3::build_exp3agent;
use crate::algorithms::thompson_sampling::ts_policy;
use crate::algorithms::DMED;
use crate::algorithms::ucb::ucb_policy;
use crate::algorithms::successive_elimination_policy::successive_elimination_policy;
use crate::algorithms::lucb::lucb_policy;
use crate::algorithms::utils::{build_reward_history, plot_data, RewardHistory, d_gaussian, d_bernoulli};
use crate::bandit::{AdversarialBanditMachine, BanditMachine, build_adversarial_bandit_machine};
mod bandit;
mod algorithms;


fn prob_bandit(){
    // params
    let T: u32 = 100;
    let rew_sigma: f64 = 0.3;

    // codes
    let mus: Vec<f64> = vec![0.5, 1.0, 2.0, 3.0];
    let machine = bandit::ProbabilisticBanditMachine{
        mus,
        arm: Box::new(bandit::build_gaussian_reward(rew_sigma)),
    };

    let mut rew_history: RewardHistory = build_reward_history(4);
    let mut miss_probs: Vec<f64> = vec![0.0_f64;0];
    let mut rew_sum: f64 = 0.0;

    let gaussian_model: Box<dyn Fn(f64, f64) -> f64 + '_>
        = Box::new(move |mu1: f64, mu2: f64| -> f64 { d_gaussian(mu1, rew_sigma.clone(), mu2, rew_sigma.clone())});
    let bernoulli_model = Box::new(d_bernoulli);
    let mut DMED_agent = DMED::build_DMEDAgent(4, gaussian_model);

    for t in 0..T {
        // let selected_arm: u32 = epsilon_greedy_policy(&rew_history, 0.1_f64);
        // let selected_arm: u32 = ucb_policy(&rew_history, t);
        // let selected_arm: u32 = ts_policy(&rew_history, rew_sigma, 0.0, 2.0);
        let selected_arm: u32 = DMED_agent.policy(t as f64, &rew_history);
        let rew = machine.get_reward(selected_arm);
        rew_history.observe(selected_arm, rew);
        let miss_prob = ((t+1-(rew_history.rewards[3].len() as u32)) as f64)/ ((t+1) as f64);

        println!("selected arm is {}, rew is {}, miss_probs {}", selected_arm, rew, miss_prob);
        miss_probs.push(miss_prob);
        rew_sum += rew;
    }
    println!("total rew is {}, miss prob is {}", rew_sum, miss_probs.last().unwrap());
    // let x_data = &(0..T).map(|x| x as f64).collect::<Vec<f64>>();
    // plot_data(&x_data, &miss_probs, "test.png", "miss probs");
}

fn beta(n: f64, delta: f64) -> f64{  // for best arm identification
    (4.0 * 4.0 * n * n / delta).ln()  // log(4*K*n^2 / delta)
}

fn best_arm_identification(){
    // params
    let rew_sigma: f64 = 0.1;
    let eps: f64 = 0.1;
    let delta: f64 = 0.01;

    // codes
    let mus: Vec<f64> = vec![0.5, 1.0, 2.8, 3.0];
    let machine = bandit::ProbabilisticBanditMachine{
        mus: mus.clone(),
        arm: Box::new(bandit::build_gaussian_reward(rew_sigma)),
    };
    let mut rew_history: RewardHistory = build_reward_history(4);
    let (optimal_arm, trial_times) = successive_elimination_policy(
        &machine, &mut rew_history, eps.clone(), delta.clone(), beta
    );
    let mut rew_history_lucb: RewardHistory = build_reward_history(4);
    let (optimal_arm_lucb, trial_times_lucb) = lucb_policy(
        &machine, &mut rew_history_lucb, eps.clone(), delta.clone(), beta
    );
    println!("optimal arm is {}, trial times is {}", optimal_arm, trial_times);
    for (id, rew_vec) in rew_history.rewards.iter().enumerate(){
        println!("arm{}: trial times {}", id, rew_vec.len());
    }
    println!("LUCB: optimal arm is {}, trial times is {}", optimal_arm_lucb, trial_times_lucb);
    for (id, rew_vec) in rew_history_lucb.rewards.iter().enumerate(){
        println!("arm{}: trial times {}", id, rew_vec.len());
    }
    let mut h_eps = 1.0 / (mus[3] - mus[2] as f64 + eps);
    for &mu in mus[0..3].iter(){
        h_eps += 1.0 / (mus[3] - mu as f64 + eps);
    }
    h_eps /= 2.0;
    println!("sample complexity is {}, {}", h_eps, h_eps * 256.0 * (4.0 * 4.0 /delta).ln());
}


fn adversarial_bandit_problem(){
    let T: u32 = 200;
    let eta = 0.2;
    let agent_gamma = 0.9;
    let bandit_gamma = 0.4;

    let mut bandit = build_adversarial_bandit_machine(4, bandit_gamma);
    let mut exp3_agent = build_exp3agent(bandit.arm_size(), agent_gamma, eta);

    let mut miss_prob: f64 = 0.0;
    let mut rew_sum: f64 = 0.0;
    for t in 1..T+1 {
        let select_arm: u32 = exp3_agent.select_arm();
        let reward: f64 = bandit.get_reward(select_arm);
        miss_prob = (miss_prob * (t-1) as f64 + (select_arm != bandit.best_arm()) as i32 as f64) / t as f64;
        rew_sum += reward;
        println!("reward {:.5}, miss prob {:.5}, rew vec {:?}", reward, miss_prob, bandit.get_rewards());
        exp3_agent.observe(select_arm, reward);
        bandit.observe(select_arm);
    }
    println!("reward sum {:.5}, miss prob {:.5}", rew_sum, miss_prob);
}

fn main(){
    prob_bandit();
    // best_arm_identification();
    // adversarial_bandit_problem();
}