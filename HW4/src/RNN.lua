require "src.Logger"

local RNN = torch.class("RNN")

function RNN:__init(inputDimension, hiddenDimension)
	logger:debug("Initializing RNN Layer")
	local fan_in = inputDimension + hiddenDimension
	local fan_out = hiddenDimension

	-- Xavier Initialization
	local stdv = math.sqrt(2 / (fan_in + fan_out))
	self.W = torch.Tensor(fan_in, fan_out):uniform(-stdv, stdv)
	self.B = torch.Tensor(fan_out, 1):uniform(-stdv, stdv)

	self.clock = 0
	self.timestamp = 0
	self._output = {}
	self._output[0] = torch.zeros(fan_out, 1)
	self._gradW = {}
	self._gradB = {}
	self.output = self._output[self.timestamp]
	self.gradW = self._gradW[self.timestamp]
	self.gradB = self._gradB[self.timestamp]
	self.gradInput = torch.Tensor()
end

function RNN:forward(input)
	self.clock = self.clock + 1
	self.timestamp = self.timestamp + 1
	-- logger:debug("Current clock: " .. self.clock)
	-- logger:debug("Current timestamp (forward): " .. self.timestamp)

	self._output[self.timestamp] = (self.W:t() * (torch.cat(self._output[self.timestamp - 1], input, 1)) + self.B):apply(math.tanh)
	self.output = self._output[self.timestamp]
	return self.output
end

function RNN:backward(input, gradOutput, scale)
	--[[
			table.remove(tbl, index) pops the element from the table tbl
			but takes the index values from 1 onwards. It can't remove
			element at index 0
			We are using index 0 to hold the first input to the network.
			So when self.timestamp reaches 1, we pop the element at index 0
			manually and copy the element at index 1 to index 0 to be used
			in the network
	  ]]
	local previousInput
	if self.timestamp > 1 then
		x = self._output[self.timestamp - 1]
		previousInput = table.remove(self._output, self.timestamp - 1)
	elseif self.timestamp == 1 then
		previousInput = self._output[0]
		self._output[0] = table.remove(self._output, 1)
	else
		logger:error("Timestamp is already 0. Can't go further back.")
		error()
	end

	local derivative_of_tanh = (1 - torch.pow(previousInput, 2))
	self._gradW[self.timestamp] = torch.cat(previousInput, input, 1) * derivative_of_tanh:t()
	self._gradB[self.timestamp] = derivative_of_tanh
	self.gradInput = self.W * derivative_of_tanh
	self.gradW = self._gradW[self.timestamp]
	self.gradB = self._gradB[self.timestamp]

	self.timestamp = self.timestamp - 1
	-- logger:debug("Current clock: " .. self.clock)
	-- logger:debug("Current timestamp (backward): " .. self.timestamp)
	return self.gradInput
end

function RNN:updateParameters(learningRate)
	for i = 1, #self._gradW do
		self.W = self.W - learningRate * self._gradW[i]
		self.B = self.B - learningRate * self._gradB[i]
	end
	self._gradW = {}
	self._gradB = {}
end
