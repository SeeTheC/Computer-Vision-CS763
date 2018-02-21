
require "xlua"
cmd = torch.CmdLine()

local cmd = torch.CmdLine()
if not opt then
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Options:')
   cmd:option("-modelName" ,"" ,"/path/to/<model>")
   cmd:option("-data " ,"data.bin" ,"/path/to/data.bin")
   cmd:option("-target" ,"labels.bin" ,"/path/to/target/labels.bin")
   cmd:text()
   opt = cmd:parse(arg or {})
end
