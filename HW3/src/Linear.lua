require "Logger"

local Linear = torch.class('Linear')

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
	self.gradInput = torch.zeros(input:size())
	for i = 1, input:size(1) do
		self.gradInput[i] = gradOutput[i]:reshape(1, self.n_output) * self.W
	end
	
	for k = 1, input:size(1) do
		local dodw = torch.Tensor(self.n_output, self.n_input * self.n_output)
		local st = 1
		for i = 1, self.n_output do
			for j = 1, self.n_input do
				dodw[i][st] = input[k][j]
				st = st + 1
			end
		end
		self.gradW[k] = (gradOutput:reshape(1, self.n_output) * dodw):reshape(self.n_output, self.n_input)
	end
	-- TODO: self.gradB
	return self.gradInput
end
