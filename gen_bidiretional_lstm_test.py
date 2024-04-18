import torch
from tools import array_to_str


batch = 1
seq_length = 4
input_dim = 2
hidden_dim = 3

model = torch.nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
rand_input = torch.Tensor(
    [
        float(f"{each:.3f}")
        for each in torch.randn(batch, seq_length, input_dim).numpy().flatten().tolist()
    ]
).reshape((batch, seq_length, input_dim))
h0 = torch.Tensor(
    [
        float(f"{each:.3f}")
        for each in torch.randn(2, batch, hidden_dim).numpy().flatten().tolist()
    ]
).reshape((2, batch, hidden_dim))
c0 = torch.Tensor(
    [
        float(f"{each:.3f}")
        for each in torch.randn(2, batch, hidden_dim).numpy().flatten().tolist()
    ]
).reshape((2, batch, hidden_dim))


def get_gate_weights(param):
    # It's a must to call transpose(), because burn's inner representation
    # of weights is [input_size, output_size]
    w1 = param.detach().numpy()[:hidden_dim].transpose(1, 0)
    w2 = param.detach().numpy()[hidden_dim : hidden_dim * 2].transpose(1, 0)
    w3 = param.detach().numpy()[hidden_dim * 2 : hidden_dim * 3].transpose(1, 0)
    w4 = param.detach().numpy()[hidden_dim * 3 :].transpose(1, 0)

    # input_gate, forget_gate, cell_gate, output_gate
    return (
        array_to_str(w1),
        array_to_str(w2),
        array_to_str(w3),
        array_to_str(w4),
    )


def get_gate_biases(param):
    # input_gate, forget_gate, cell_gate, output_gate
    return (
        array_to_str(param.detach().numpy()[:hidden_dim]),
        array_to_str(param.detach().numpy()[hidden_dim : hidden_dim * 2]),
        array_to_str(param.detach().numpy()[hidden_dim * 2 : hidden_dim * 3]),
        array_to_str(param.detach().numpy()[hidden_dim * 3 :]),
    )


for name, param in model.named_parameters():
    shape = param.shape
    new_value = torch.Tensor(
        [float(f"{each:.3f}") for each in param.flatten().detach().numpy().tolist()]
    ).reshape(shape)
    param.data = new_value

    if name == "weight_ih_l0":
        (
            input_gate_input_weights,
            forget_gate_input_weights,
            cell_gate_input_weights,
            output_gate_input_weights,
        ) = get_gate_weights(param)

    if name == "weight_hh_l0":
        (
            input_gate_hidden_weights,
            forget_gate_hidden_weights,
            cell_gate_hidden_weights,
            output_gate_hidden_weights,
        ) = get_gate_weights(param)

    if name == "bias_ih_l0":
        (
            input_gate_input_biases,
            forget_gate_input_biases,
            cell_gate_input_biases,
            output_gate_input_biases,
        ) = get_gate_biases(param)

    if name == "bias_hh_l0":
        (
            input_gate_hidden_biases,
            forget_gate_hidden_biases,
            cell_gate_hidden_biases,
            output_gate_hidden_biases,
        ) = get_gate_biases(param)

    if name == "weight_ih_l0_reverse":
        (
            input_gate_input_weights_reverse,
            forget_gate_input_weights_reverse,
            cell_gate_input_weights_reverse,
            output_gate_input_weights_reverse,
        ) = get_gate_weights(param)

    if name == "weight_hh_l0_reverse":
        (
            input_gate_hidden_weights_reverse,
            forget_gate_hidden_weights_reverse,
            cell_gate_hidden_weights_reverse,
            output_gate_hidden_weights_reverse,
        ) = get_gate_weights(param)

    if name == "bias_ih_l0_reverse":
        (
            input_gate_input_biases_reverse,
            forget_gate_input_biases_reverse,
            cell_gate_input_biases_reverse,
            output_gate_input_biases_reverse,
        ) = get_gate_biases(param)

    if name == "bias_hh_l0_reverse":
        (
            input_gate_hidden_biases_reverse,
            forget_gate_hidden_biases_reverse,
            cell_gate_hidden_biases_reverse,
            output_gate_hidden_biases_reverse,
        ) = get_gate_biases(param)


forward_result_with_init_state, (hn_with_init_state, cn_with_init_state) = model(rand_input, (h0, c0))
forward_result_without_init_state, (hn_without_init_state, cn_without_init_state) = model(rand_input)

res_str = f"""
#[test]
fn test_bidirectional() {{
    TestBackend::seed(0);
    let config = BiLstmConfig::new({input_dim}, {hidden_dim}, true);
    let device = Default::default();
    let mut lstm = config.init(&device);

    fn create_gate_controller<const D1: usize, const D2: usize>(
        input_weights: [[f32; D1]; D2],
        input_biases: [f32; D1],
        hidden_weights: [[f32; D1]; D1],
        hidden_biases: [f32; D1],
        device: &Device<TestBackend>,
    ) -> GateController<TestBackend> {{
        let d_input = input_weights[0].len();
        let d_output = input_weights.len();

        let input_record = LinearRecord {{
            weight: Param::from_data(Data::from(input_weights), device),
            bias: Some(Param::from_data(Data::from(input_biases), device)),
        }};
        let hidden_record = LinearRecord {{
            weight: Param::from_data(Data::from(hidden_weights), device),
            bias: Some(Param::from_data(Data::from(hidden_biases), device)),
        }};
        GateController::create_with_weights(
            d_input,
            d_output,
            true,
            Initializer::XavierUniform {{ gain: 1.0 }},
            input_record,
            hidden_record,
        )
    }}

    let input = Tensor::<TestBackend, 3>::from_data(
        Data::from({array_to_str(rand_input.detach().numpy())}),
        &device,
    );
    let h0 = Tensor::<TestBackend, 3>::from_data(
        Data::from({array_to_str(h0.detach().numpy())}),
        &device,
    );
    let c0 = Tensor::<TestBackend, 3>::from_data(
        Data::from({array_to_str(c0.detach().numpy())}),
        &device,
    );

    lstm.forward.input_gate = create_gate_controller(
        {input_gate_input_weights},
        {input_gate_input_biases},
        {input_gate_hidden_weights},
        {input_gate_hidden_biases},
        &device,
    );

    lstm.forward.forget_gate = create_gate_controller(
        {forget_gate_input_weights},
        {forget_gate_input_biases},
        {forget_gate_hidden_weights},
        {forget_gate_hidden_biases},
        &device,
    );

    lstm.forward.cell_gate = create_gate_controller(
        {cell_gate_input_weights},
        {cell_gate_input_biases},
        {cell_gate_hidden_weights},
        {cell_gate_hidden_biases},
        &device,
    );

    lstm.forward.output_gate = create_gate_controller(
        {output_gate_input_weights},
        {output_gate_input_biases},
        {output_gate_hidden_weights},
        {output_gate_hidden_biases},
        &device,
    );

    lstm.reverse.input_gate = create_gate_controller(
        {input_gate_input_weights_reverse},
        {input_gate_input_biases_reverse},
        {input_gate_hidden_weights_reverse},
        {input_gate_hidden_biases_reverse},
        &device,
    );

    lstm.reverse.forget_gate = create_gate_controller(
        {forget_gate_input_weights_reverse},
        {forget_gate_input_biases_reverse},
        {forget_gate_hidden_weights_reverse},
        {forget_gate_hidden_biases_reverse},
        &device,
    );

    lstm.reverse.cell_gate = create_gate_controller(
        {cell_gate_input_weights_reverse},
        {cell_gate_input_biases_reverse},
        {cell_gate_hidden_weights_reverse},
        {cell_gate_hidden_biases_reverse},
        &device,
    );

    lstm.reverse.output_gate = create_gate_controller(
        {output_gate_input_weights_reverse},
        {output_gate_input_biases_reverse},
        {output_gate_hidden_weights_reverse},
        {output_gate_hidden_biases_reverse},
        &device,
    );

    let expected_output_with_init_state = Data::from({array_to_str(forward_result_with_init_state.detach().numpy(), 5)});
    let expected_output_without_init_state = Data::from({array_to_str(forward_result_without_init_state.detach().numpy(), 5)});
    let expected_hn_with_init_state = Data::from({array_to_str(hn_with_init_state.detach().numpy(), 5)});
    let expected_cn_with_init_state = Data::from({array_to_str(cn_with_init_state.detach().numpy(), 5)});
    let expected_hn_without_init_state = Data::from({array_to_str(hn_without_init_state.detach().numpy(), 5)});
    let expected_cn_without_init_state = Data::from({array_to_str(cn_without_init_state.detach().numpy(), 5)});

    let (output_with_init_state, state_with_init_state) = lstm.forward(input.clone(), Some(LstmState::new(c0, h0)));
    let (output_without_init_state, state_without_init_state) = lstm.forward(input, None);

    output_with_init_state.to_data().assert_approx_eq(&expected_output_with_init_state, 3);
    output_without_init_state.to_data().assert_approx_eq(&expected_output_without_init_state, 3);
    state_with_init_state.hidden.to_data().assert_approx_eq(&expected_hn_with_init_state, 3);
    state_with_init_state.cell.to_data().assert_approx_eq(&expected_cn_with_init_state, 3);
    state_without_init_state.hidden.to_data().assert_approx_eq(&expected_hn_without_init_state, 3);
    state_without_init_state.cell.to_data().assert_approx_eq(&expected_cn_without_init_state, 3);
}}
""".strip()

with open("bidirectional_lstm.txt", "w") as f:
    f.write(res_str)
