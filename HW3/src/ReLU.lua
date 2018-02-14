require "Logger"

local ReLU = torch.class('ReLU')

function ReLU:__init(leak)
	logger:debug("Initializing ReLU Layer")

	-- Leaky ReLU
	self.leak = leak or 0
	
	-- No need to allocate memory to these tensors
	-- They will be replaced by the calculated values
	self.output = torch.Tensor()
	self.gradInput = torch.Tensor()
end

function ReLU:forward(input)
	self.output = torch.zeros(input:size())
	for i = 1, input:size(1) do
		self.output[i] = (input[i] > 0) and input[i] or self.leak * input[i]
	end
	return self.output
end

function ReLU:backward(input, gradOutput)
	self.gradInput = torch.zeros(input:size())
	for i = 1, input:size(1) do
		self.gradInput[i] = gradOutput[i] * (input[i] > 0 and 1 or self.leak)
	end
	return self.gradInput
end
