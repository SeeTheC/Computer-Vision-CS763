require "Logger"

local Linear = torch.class("Linear")

function Linear:__init(n_input, n_output)
	logger:debug("Initializing Linear Layer")

	-- Input parameters and initialization
	self.n_input = n_input
	self.n_output = n_output
	self.W = torch.randn(n_output, n_input)
	self.B = torch.randn(n_output)

	-- No need to allocate memory to these tensors
	-- They will be replaced by the calculated values
	self.output = torch.Tensor()
	self.gradW = torch.Tensor()
	self.gradB = torch.Tensor()
	self.gradInput = torch.Tensor()
end

function Linear:forward(input)
	self.output = torch.zeros(input:size(1), self.n_output)
	for i = 1, input:size(1) do
		self.output[i] = self.W * input[i] + self.B
	end
	return self.output
end

function Linear:backward(input, gradOutput)
	self.gradInput = gradOutput * self.W

	local gradOutput_T = gradOutput:t()
	self.gradW = gradOutput_T * input

	self.gradB = torch.sum(gradOutput_T, 2)

	return self.gradInput
end

function Linear:resetGrads()
	self.gradW = torch.Tensor()
	self.gradB = torch.Tensor()
	self.gradInput = torch.Tensor()
end

function Linear:updateParams(learning_rate)
	self.W = self.W - learning_rate * self.gradW
	self.B = self.B - learning_rate * self.gradB
end
