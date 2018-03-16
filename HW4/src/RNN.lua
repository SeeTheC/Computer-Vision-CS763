require "src.Logger"

local RNN = torch.class("RNN")

function RNN:__init(inputDimension, hiddenDimension)
	logger:debug("Initializing RNN Layer")
	self.inputDimension = inputDimension
	self.hiddenDimension = hiddenDimension

	-- Xavier Initialization
	self.fan_in = self.inputDimension + self.hiddenDimension
	self.fan_out = self.hiddenDimension
	local stdv = math.sqrt(2.0 / (self.fan_in + self.fan_out))
	self.W = torch.Tensor(self.fan_in, self.fan_out):uniform(-stdv, stdv)
	self.B = torch.Tensor(self.fan_out, 1):uniform(-stdv, stdv)

	self.gradientClipThreshold = 1e2

	self:resetGradsAndOutputs()
end

function RNN:resetGradsAndOutputs()
	self._hiddenOutput = {}
	self._hiddenOutput[0] = torch.zeros(self.fan_out, 1)
	self.output = nil
	self.gradW = torch.zeros(self.W:size())
	self.gradB = torch.zeros(self.B:size())
	self.gradInput = torch.Tensor()
end

function RNN:forward(input)
	local size = input:size(2)
	for i = 1, size do
		self._hiddenOutput[i] = torch.tanh(self.W:t() * (torch.cat(self._hiddenOutput[i - 1], input:select(2, i), 1)) + self.B)
	end
	self.output = self._hiddenOutput[size]
	return self.output
end

function RNN:backward(input, gradOutput)
	self.gradW = torch.zeros(self.W:size())
	self.gradB = torch.zeros(self.B:size())
	local nextGradOutput = gradOutput
	for i = input:size(2), 1, -1 do
		local derivativeOfTanh = torch.cmul(gradOutput, 1 - torch.pow(self._hiddenOutput[i], 2))

		-- Gradient Clipping
		local gradW = torch.cat(self._hiddenOutput[i - 1], input:select(2, i), 1) * derivativeOfTanh:t()
		local gradWNorm = torch.sum(torch.pow(gradW, 2))
		if gradWNorm > self.gradientClipThreshold then
			logger:warn("Norm of gradient of W went above the threshold: " .. gradWNorm)
			gradW = gradW * (self.gradientClipThreshold / gradWNorm)
		end
		self.gradW = self.gradW + gradW

		self.gradB = self.gradB + derivativeOfTanh
		nextGradOutput = self.W:narrow(1, 1, self.fan_out):t() * derivativeOfTanh
	end
	self.gradInput = nextGradOutput
	return self.gradInput
end

function RNN:updateParameters(learningRate)
	self.W = self.W - learningRate * self.gradW
	self.B = self.B - learningRate * self.gradB
end

function RNN:__tostring__()
	return torch.type(self) .. string.format(': Hidden Dimension = %d | Input Dimension = %d', self.hiddenDimension, self.inputDimension)
end
