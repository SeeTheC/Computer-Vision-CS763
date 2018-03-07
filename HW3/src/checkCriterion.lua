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

local input = torch.load(opt["i"])
local target = torch.load(opt["t"])

local criterion = Criterion()
local output = criterion:forward(input, target)
local gradOutput = criterion:backward(input, target)

torch.save(opt["og"], gradOutput)
