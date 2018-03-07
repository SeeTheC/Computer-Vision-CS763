require "Logger"

local BatchNormalization = torch.class("BatchNormalization")

function BatchNormalization:__init()
	logger:debug("Initializing BatchNormalization Layer")
	self.output = torch.Tensor()
end

function BatchNormalization:forward(input)
	self.output = input:clone()
	if self.output:nDimension() == 1 then
		local mean = torch.mean(self.output)
		local stddev = torch.std(self.output)

		self.output = (self.output - mean) / stddev
	else
		local mean = torch.mean(self.output, 1)
		local stddev = torch.std(self.output, 1)

		for i = 1, self.output:size(1) do
			self.output[i] = torch.cdiv((self.output[i] - mean), stddev)
		end
	end

	return self.output
end

function BatchNormalization:backward(input, gradOutput)
	return gradOutput
end

function BatchNormalization:resetGrads()
	return
end

function BatchNormalization:__tostring__()
	return torch.type(self)
end
