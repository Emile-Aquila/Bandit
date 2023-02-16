use rand_distr::Distribution;

pub trait BanditMachine{
    fn get_reward(&self, arm_index: u32) -> f64;
    // indexに対応するアームを引く
}

pub trait ProbabilisticBanditArm {
    fn get_reward(&self, mu: f64) -> f64;
}


pub struct ProbabilisticBanditMachine {
    pub(crate) mus: Vec<f64>,
    pub(crate) arm: Box<dyn ProbabilisticBanditArm>,
}

impl BanditMachine for ProbabilisticBanditMachine{
    fn get_reward(&self, arm_index: u32) -> f64 {
        self.arm.get_reward(self.mus[arm_index as usize])
    }
}

pub struct GaussianReward{
    dist_normal: rand_distr::Normal<f64>,
}

impl ProbabilisticBanditArm for GaussianReward{
    fn get_reward(&self, mu: f64) -> f64 {
        self.dist_normal.sample(&mut rand::thread_rng()) + mu
    }
}

pub fn build_gaussian_reward(sigma: f64) -> GaussianReward{
    GaussianReward {
        dist_normal: rand_distr::Normal::new(0.0, sigma).unwrap(),
    }
}

