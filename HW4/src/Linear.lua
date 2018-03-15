require "src.Logger"

local Linear = torch.class("Linear")

function Linear:__init(fan_in, fan_out)
	logger:debug("Initializing Linear Layer")

	self.fan_in = fan_in
	self.fan_out = fan_out
	-- Xavier Initialization
	local stdv = math.sqrt(2 / (self.fan_in + self.fan_out))
	self.W = torch.Tensor(self.fan_out, self.fan_in):uniform(-stdv, stdv)
	self.B = torch.Tensor(self.fan_out):uniform(-stdv, stdv)

	self:resetGradsAndOutputs()
end

function Linear:resetGradsAndOutputs()
	self.output = torch.Tensor()
	self.gradW = torch.Tensor()
	self.gradB = torch.Tensor()
	self.gradInput = torch.Tensor()
end

function Linear:forward(input)
	self.output = self.W * input + self.B
	return self.output
end

function Linear:backward(input, gradOutput)
	self.gradInput = self.W:t() * gradOutput
	self.gradW = gradOutput * input:t()
	self.gradB = gradOutput
	return self.gradInput
end

function Linear:updateParameters(learningRate)
	self.W = self.W - learningRate * self.gradW
	self.B = self.B - learningRate * self.gradB
end

function Linear:__tostring__()
	return torch.type(self) .. string.format(': (%d -> %d)', self.fan_in, self.fan_out) .. " with bias"
end
