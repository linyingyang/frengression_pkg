"""Regression checks for the history passed to sequential covariate generators."""

import unittest

import torch

from frengression.frengression import FrengressionSeq


class RecordingGenerator(torch.nn.Module):
    def __init__(self, step, output_dim):
        super().__init__()
        self.step = step
        self.output_dim = output_dim
        self.inputs = []
        self.outputs = []

    def forward(self, inputs):
        self.inputs.append(inputs.clone())
        values = torch.arange(self.output_dim, dtype=inputs.dtype, device=inputs.device)
        rows = torch.arange(inputs.shape[0], dtype=inputs.dtype, device=inputs.device)
        output = 100 * rows[:, None] + 10 * self.step + values[None, :]
        self.outputs.append(output.clone())
        return output


class SequentialHistoryTests(unittest.TestCase):
    def make_model(self, time_steps):
        model = FrengressionSeq(
            x_dim=2, y_dim=1, z_dim=3, T=time_steps, s_dim=2,
            num_layer=1, hidden_dim=4, noise_dim=1,
            device=torch.device("cpu"),
        )
        model.model_xz = [RecordingGenerator(t, 5) for t in range(time_steps)]
        return model

    def test_generated_history_contains_every_previous_time_point(self):
        for time_steps in (1, 2, 5):
            with self.subTest(time_steps=time_steps):
                model = self.make_model(time_steps)
                baseline = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
                x, z = model.sample_xz(s=baseline)
                torch.testing.assert_close(model.model_xz[0].inputs[0], baseline)
                for t in range(1, time_steps):
                    expected = torch.cat([baseline, x[:, :2 * t], z[:, :3 * t]], dim=1)
                    torch.testing.assert_close(model.model_xz[t].inputs[0], expected)
                for t in range(time_steps):
                    output = model.model_xz[t].outputs[0]
                    torch.testing.assert_close(x[:, 2 * t:2 * (t + 1)], output[:, :2])
                    torch.testing.assert_close(z[:, 3 * t:3 * (t + 1)], output[:, 2:])

    def test_supplied_histories_are_used_during_teacher_forcing(self):
        model = self.make_model(5)
        baseline = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        x = torch.arange(20, dtype=torch.float32).reshape(2, 10)
        z = torch.arange(30, dtype=torch.float32).reshape(2, 15) + 200
        model.sample_xz(s=baseline, x=x, z=z)
        torch.testing.assert_close(model.model_xz[0].inputs[0], baseline)
        for t in range(1, 5):
            expected = torch.cat([baseline, x[:, :2 * t], z[:, :3 * t]], dim=1)
            torch.testing.assert_close(model.model_xz[t].inputs[0], expected)


if __name__ == "__main__":
    unittest.main()
