require "Logger"

local SpatialConvolution = torch.class('SpatialConvolution')


function SpatialConvolution:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
	dW = dW or 1
	dH = dH or 1

	self.nInputPlane = nInputPlane
	self.nOutputPlane = nOutputPlane
	self.kW = kW
	self.kH = kH

	self.dW = dW
	self.dH = dH
	self.padW = padW or 0
	self.padH = padH or self.padW

	self.weight = torch.Tensor(nOutputPlane, nInputPlane*kH*kW)
	self.bias = torch.Tensor(nOutputPlane)
	self.gradWeight = torch.Tensor(nOutputPlane, nInputPlane*kH*kW)
	self.gradBias = torch.Tensor(nOutputPlane)

end

function SpatialConvolution:forward1(input)
	rows=input:size(2)
	cols=input:size(3)
	filerSize=(self.kW)*(self.kH)
	numOfFilters=self.weight:size(1)
	local noutputRow=math.floor((rows-self.kW)/self.dW)+1
	local noutputCol=math.floor((cols-self.kH)/self.dH)+1
	local output=torch.Tensor(self.nOutputPlane,noutputRow,noutputCol)
	--padding the image
	--Performing Convolution with strides dW and dH
	for i=1,rows-self.kH+1 do
		for j=1,cols-self.kW+1 do
			for op=1,self.nOutputPlane do
				local temp=0
				local k=1 -- for a particular wt column
				for ip=1,self.nInputPlane do
					for r=i,self.kH+i-1 do
						for c=j,self.kW+j-1 do
							temp=temp+input[ip][r][c]*self.weight[op][k]
							k=k+1
						end
					end
				output[op][i][j]=temp+self.bias[op]
				end
			end
			j=j+self.dW-1--to increase the step size for strides
		end
		i=i+self.dH-1
	end
	return output
end

function SpatialConvolution:forward(input)
	--padding the image
	local rows=input:size(2)
	local cols=input:size(3)
	local sFilter=self.kW*self.kH--for single input layer
	filterSize=self.weight:size(2)--for all the input layers
	numOfFilters=self.weight:size(1)
	local nRow=math.floor((rows-self.kW)/self.dW)+1
	local nCol=math.floor((cols-self.kH)/self.dH)+1
	local inputVector=torch.Tensor(filterSize,nRow*nCol)
	local outputVector=torch.Tensor(numOfFilters,nRow*nCol)
	local output=torch.Tensor(self.nOutputPlane,nRow,nCol)
	--Performing Convolution with strides dW and dH
	local k=1
	for i=1,rows-self.kH+1,self.dH do
		for j=1,cols-self.kW+1,self.dW do
			for nIP=1,self.nInputPlane do
				--print(i)
				inputVector[{{(nIP-1)*sFilter+1,nIP*sFilter},{k}}]=input[{{nIP},{i,i+self.kH-1},{j,j+self.kW-1}}]:reshape(sFilter,1)
			end
			k=k+1
		end
	end
	outputVector=self.weight*inputVector
	for i=1,self.nOutputPlane do
		output[{{i},{1,nRow},{1,nCol}}]=outputVector[i]:reshape(nRow,nCol)+self.bias[i]
	end
	return output
end


function SpatialConvolution:backward(input, gradOutput)
	--padding the image
	--here gOutput is acting as weight in forward pass
	local rows=input:size(2)
	local cols=input:size(3)
	grows=gradOutput:size(2)
	gcols=gradOutput:size(3)
	local sFilter=self.kH*self.kW-- size of gradient weight for single input layer
	--Performing Convolution with strides dW and dH
	local l=1
	for i=1,rows-(grows+self.dH)+2 do
		for j=1,cols-(gcols+self.dW)+2 do

			for nOP=1,self.nOutputPlane do
				for nIP=1,self.nInputPlane do
					local temp=0
					for m=0,gcols-1 do
						for n=0,grows-1 do
							--print(1)
							temp=temp+(input[nIP][i+m*self.dW][j+n*self.dH]*gradOutput[nOP][m+1][n+1])
							--print(2)
						end
					end
					self.gradWeight[nOP][(nIP-1)*sFilter+l]=temp
					--print(nOP,(nIP-1)*sFilter+l)
				end
			end
			l=l+1
		end
	end
	--works only for stride =1
	inputVector=torch.Tensor(self.nOutputPlane,self.nInputPlane,grows*gcols,rows*cols)
	gradInput=torch.zeros(rows*cols,self.nOutputPlane)
	temp2=torch.zeros(1,rows*cols)
	for nOP=1,self.nOutputPlane do
		for nIP=1,self.nInputPlane do
			local k=1
			temp=torch.zeros(1,rows*cols)
			for t=1,self.kH do
				temp[{{1},{k,self.kW+k-1}}]=self.weight[{{nOP},{((nIP-1)*self.kW*self.kH+(t-1)*self.kW+1),(self.kW*t+(nIP-1)*self.kW*self.kH)}}]
				k=k+self.kW
				temp[{{1},{k,(rows-self.kW)+k-1}}]=0
				k=k+(rows-self.kW)
			end
			temp2=temp
			local i=0
			for r=1,gcols*grows do
				inputVector[{{nOP},{nIP},{r},{1,rows*cols}}]=temp2
				i=i+1
				temp2=torch.zeros(1,rows*cols)
				temp2[{{1},{r+self.dW+torch.floor(i/grows),rows*cols}}]=temp[{{1},{1,rows*cols-self.dW-r-torch.floor(i/grows)+1}}]
				--make changes here to increase the stride
			end
		end
	end
	gradOutputVector=torch.Tensor(self.nOutputPlane,grows*gcols)
	for i=1,self.nOutputPlane do
		gradOutputVector[i]=gradOutput[{{i},{1,grows},{1,gcols}}]:reshape(grows*gcols,1)
	end
	temp=torch.zeros(grows*gcols,rows*cols)
	for i=1,self.nInputPlane do
		for j=1,self.nOutputPlane do
			temp[{{1,grows*gcols},{1,rows*cols}}]=inputVector[{{j},{i},{1,grows*gcols},{1,rows*cols}}]
			--print(temp)
			--print(gradOutputVector)
			gradInput[{{1,rows*cols},{i}}]=gradInput[{{1,rows*cols},{i}}]+gradOutputVector[{{j},{1,grows*gcols}}]*temp
		end
	end
	for i=1,self.nOutputPlane do
		self.gradBias[i] = torch.sum(gradOutput[{{i},{1,grows},{1,gcols}}])
	end
	--self.gradBias = torch.sum(gradOutput_T, 2)
	return gradInput:t():reshape(self.nInputPlane,rows,cols)
end
