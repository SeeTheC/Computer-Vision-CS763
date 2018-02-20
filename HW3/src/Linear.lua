require "Logger"

local Linear = torch.class("Linear")

function Linear:__init(n_input, n_output)
	logger:debug("Initializing Linear Layer")

	-- Input parameters and initialization
	self.n_input = n_input
	self.n_output = n_output
	local stdv = 1./math.sqrt(self.n_input)
	self.W = torch.Tensor(n_output, n_input):uniform(-stdv, stdv)
	self.B = torch.Tensor(n_output):uniform(-stdv, stdv)

	-- No need to allocate memory to these tensors
	-- They will be replaced by the calculated values
	self.output = torch.Tensor()
	self.gradW = torch.zeros(n_output, n_input)
	self.gradB = torch.zeros(n_output)
	self.gradInput = torch.Tensor()
end

function Linear:forward(input)
	local nElement = self.output:nElement()
	local batch_size = input:size(1)
	self.output:resize(batch_size, self.n_output)
	if self.output:nElement() ~= nElement then
		self.output:zero()
	end
	self.output = torch.zeros(batch_size, self.n_output)
	for i = 1, batch_size do
		self.output[i] = self.W * input[i] + self.B
	end
	return self.output
end

function Linear:backward(input, gradOutput, learningRate)
	learningRate = learningRate or 1

	local nElement = self.gradInput:nElement()
	self.gradInput:resizeAs(input)
	if self.gradInput:nElement() ~= nElement then
		self.gradInput:zero()
	end

	self.gradInput = gradOutput * self.W

	local gradW = gradOutput:t() * input

	local gradB = torch.sum(gradOutput:t(), 2)

	self:updateParams(learningRate, gradW, gradB)
	return self.gradInput
end

function Linear:resetGrads()
	self.gradW = torch.zeros(self.n_output, self.n_input)
	self.gradB = torch.zeros(self.n_output)
	self.gradInput = torch.Tensor()
end

function Linear:updateParams(learningRate, gradW, gradB)
	local updateW = learningRate * (0.1 * gradW + 0.9 * self.gradW)
	local updateB = learningRate * (0.1 * gradB + 0.9 * self.gradB)
	self.W = self.W - updateW
	self.B = self.B - updateB
	self.gradW = gradW
	self.gradB = gradB
end

function Linear:__tostring__()
	return torch.type(self) .. string.format(': (%d -> %d)', self.n_input, self.n_output) .. " with bias"
end
