use crate::algorithms::utils::{argmax, mean, RewardHistory};
use rand_distr::{Distribution, Normal};

fn gauss_posterior(rew_sigma: f64, prior_mu: f64, prior_sigma: f64, samples: &Vec<f64>) -> (f64, f64){
    let n: f64 = samples.len() as f64;
    let mu: f64 = mean(samples);
    let posterior_mu: f64 = (n*mu*prior_sigma.powf(2.0) + rew_sigma.powf(2.0)*prior_mu)
        / (n*prior_sigma.powf(2.0) + rew_sigma.powf(2.0));
    let posterior_sigma: f64 = (prior_sigma.powf(2.0) * rew_sigma.powf(2.0))
        / (n*prior_sigma.powf(2.0) + rew_sigma.powf(2.0));
    (posterior_mu, posterior_sigma)
}

pub fn ts_policy(rew_history: &RewardHistory, rew_sigma: f64, prior_mu: f64, prior_sigma: f64) -> u32{
    // 報酬のモデルに正規分布を用いたThompson Sampling Algorithm
    let mut sampled_mus = vec![0.0; 0];
    let normal = Normal::new(0.0_f64, 1.0_f64).unwrap();

    for rewards in &rew_history.rewards{
        let (post_mu, post_sigma) = gauss_posterior(rew_sigma, prior_mu, prior_sigma, rewards);
        let sampled_mu: f64 = normal.sample(&mut rand::thread_rng())*post_sigma + post_mu;
        sampled_mus.push(sampled_mu);
    }

    argmax(&sampled_mus) as u32
}