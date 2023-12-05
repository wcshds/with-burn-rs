import torch
from torch import nn
from tools import array_to_str
import datetime

# Target are to be un-padded
T = 30      # Input sequence length
C = 10      # Number of classes (including blank)
N = 1      # Batch size

# Initialize random batch of input vectors, for *size = (T, N, C)
rand_input = torch.Tensor(
    [
        float(f"{each:.3f}")
        for each in torch.randn(T, N, C).log_softmax(2).numpy().flatten().tolist()
    ]
).reshape((T, N, C)).detach().requires_grad_()
input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)
# Initialize random batch of targets (0 = blank, 1:C = classes)
target_lengths = torch.randint(low=1, high=T, size=(N,), dtype=torch.long)
target = torch.randint(low=1, high=C, size=(sum(target_lengths),), dtype=torch.long)

ctc_loss = nn.CTCLoss(reduction="sum")

start_time = datetime.datetime.now()
for i in range(1):
    loss = ctc_loss(rand_input, target, input_lengths, target_lengths)
print(datetime.datetime.now() - start_time)
print(loss)
# loss.backward()

res_str = f"""
#[test]
fn test_ctc_loss() {{
    let input = Tensor::<TestBackend, 3>::from_data({array_to_str(rand_input.detach().numpy().transpose([1, 0, 2]))});
    let target = Tensor::<TestBackend, 1, Int>::from_data({array_to_str(target.detach().numpy(), 0)});
    let input_lengths = Tensor::<TestBackend, 1, Int>::from_data({array_to_str(input_lengths.detach().numpy(), 0)});
    let target_lengths = Tensor::<TestBackend, 1, Int>::from_data({array_to_str(target_lengths.detach().numpy(), 0)});
    let expected_res = Data::from([{loss.detach().numpy()}]);

    let ctc_loss = CTCLoss::<TestBackend>::new(0);
    let res = ctc_loss.forward(
        input,
        target,
        input_lengths,
        target_lengths,
        Some(Reduction::Sum)
    );
    
    res.to_data().assert_approx_eq(&expected_res, 3);
}}
""".strip()

with open("ctc_loss.txt", 'w') as f:
    f.write(res_str)