use crate::bandit::BanditMachine;
mod bandit;


fn main(){
    let mus: Vec<f64> = vec![1.0, 2.0, 3.0, 1.0];
    let machine = bandit::ProbabilisticBanditMachine{
        mus,
        arm: Box::new(bandit::build_gaussian_reward(2.0)),
    };
    for i in 0..4 {
        println!("{}", machine.get_reward(i));
    }
}