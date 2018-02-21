
require "xlua"
cmd = torch.CmdLine()

local cmd = torch.CmdLine()
if not opt then
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Options:')
   cmd:option("-modelName" ,"" ,"/path/to/<model>")
   cmd:option("-data " ,"test.bin" ,"/path/to/test.bin (test.bin is a 4D tensor)")
   cmd:option("-target" ,"testPrediction.bin" ,"/path/to/target/testPrediction.bin")
   cmd:text()
   opt = cmd:parse(arg or {})
end
