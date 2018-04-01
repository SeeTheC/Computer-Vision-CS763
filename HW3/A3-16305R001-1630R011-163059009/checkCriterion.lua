require "xlua"
require "Criterion"

cmd = torch.CmdLine()

local cmd = torch.CmdLine()
if not opt then
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Options:')
   cmd:option("-i" ,"input.bin" ,"/path/to/input.bin")
   cmd:option("-t" ,"target.bin" ,"/path/to/target.bin")
   cmd:option("-og" ,"gradOutput.bin" ,"/path/to/gradOutput.bin")
   cmd:text()
   opt = cmd:parse(arg or {})
end

local input = torch.load(opt["i"]):double()
local target = torch.load(opt["t"]):double()
local target_onehot = torch.zeros(target:size(1), input:size(2))
for i = 1, target:size(1) do
	print(i, target[i][1])
	target_onehot[i][target[i][1]] = 1
end

local criterion = Criterion()
local output = criterion:forward(input, target_onehot)
local gradOutput = criterion:backward(input, target_onehot)

torch.save(opt["og"], gradOutput)
--
-- local x = torch.load(opt["og"])
-- print(torch.sum(gradOutput - x))
