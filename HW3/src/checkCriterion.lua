require "xlua"
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

--print(input.fields)
