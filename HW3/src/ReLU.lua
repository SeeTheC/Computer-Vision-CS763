require "Logger"

local ReLU = torch.class("ReLU")

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
	self.output = input:clone()
	self.output:apply(
		function(p)
			return (p > 0 and p or self.leak * p)
		end
	)
	return self.output
end

function ReLU:backward(input, gradOutput)
	self.gradInput = input:clone()
	self.gradInput:apply(
		function(p)
			return (p > 0 and 1 or self.leak)
		end
	)
	self.gradInput = torch.cmul(self.gradInput, gradOutput)
	return self.gradInput
end

function ReLU:resetGrads()
	self.gradInput = torch.Tensor()
end
