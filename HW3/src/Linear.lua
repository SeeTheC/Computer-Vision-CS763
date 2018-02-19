require "Logger"

local Linear = torch.class("Linear")

function Linear:__init(n_input, n_output)
	logger:debug("Initializing Linear Layer")

	-- Input parameters and initialization
	self.n_input = n_input
	self.n_output = n_output
	self.W = torch.randn(n_output, n_input) / math.sqrt(n_input / 2) * 0.01
	self.B = torch.randn(n_output) / math.sqrt(n_input / 2) * 0.01

	-- No need to allocate memory to these tensors
	-- They will be replaced by the calculated values
	self.output = torch.Tensor()
	self.gradW = torch.zeros(n_output, n_input)
	self.gradB = torch.Tensor(n_output)
	self.gradInput = torch.Tensor()
end

function Linear:forward(input)
	local nElement = self.output:nElement()
	local batch_size = input:size(1)
	self.output:resize(batch_size, self.n_output)
	if self.output:nElement() ~= nElement then
		self.output:zero()
	end
	self.output = torch.zeros(input:size(1), self.n_output)
	for i = 1, input:size(1) do
		self.output[i] = self.output[i] + self.W * input[i] + self.B
	end
	return self.output
end

function Linear:backward(input, gradOutput, scale)
	scale = scale or 1

	local nElement = self.gradInput:nElement()
	self.gradInput:resizeAs(input)
	if self.gradInput:nElement() ~= nElement then
		self.gradInput:zero()
	end

	self.gradInput = self.gradInput + gradOutput * self.W

	self.gradW = self.gradW + scale * gradOutput:t() * input

	self.gradB = self.gradB + scale * torch.sum(gradOutput:t(), 2)

	self:updateParams()
	return self.gradInput
end

function Linear:resetGrads()
	self.gradW = torch.zeros(n_output, n_input)
	self.gradB = torch.Tensor(n_output)
	self.gradInput = torch.Tensor()
end

function Linear:updateParams()
	self.W = self.W - self.gradW
	self.B = self.B - self.gradB
end

function Linear:__tostring__()
	return torch.type(self) .. string.format('(%d -> %d)', self.n_input, self.n_output)
end
