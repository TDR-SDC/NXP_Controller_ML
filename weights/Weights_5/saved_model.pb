т
пЏ
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

њ
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%Зб8"&
exponential_avg_factorfloat%  ?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
О
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718З
|
input/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameinput/kernel
u
 input/kernel/Read/ReadVariableOpReadVariableOpinput/kernel*&
_output_shapes
:*
dtype0
l

input/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
input/bias
e
input/bias/Read/ReadVariableOpReadVariableOp
input/bias*
_output_shapes
:*
dtype0

Conv_1-2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameConv_1-2/kernel
{
#Conv_1-2/kernel/Read/ReadVariableOpReadVariableOpConv_1-2/kernel*&
_output_shapes
:*
dtype0
r
Conv_1-2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameConv_1-2/bias
k
!Conv_1-2/bias/Read/ReadVariableOpReadVariableOpConv_1-2/bias*
_output_shapes
:*
dtype0

Conv_2-1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameConv_2-1/kernel
{
#Conv_2-1/kernel/Read/ReadVariableOpReadVariableOpConv_2-1/kernel*&
_output_shapes
:*
dtype0
r
Conv_2-1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameConv_2-1/bias
k
!Conv_2-1/bias/Read/ReadVariableOpReadVariableOpConv_2-1/bias*
_output_shapes
:*
dtype0

Conv_2-2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameConv_2-2/kernel
{
#Conv_2-2/kernel/Read/ReadVariableOpReadVariableOpConv_2-2/kernel*&
_output_shapes
:*
dtype0
r
Conv_2-2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameConv_2-2/bias
k
!Conv_2-2/bias/Read/ReadVariableOpReadVariableOpConv_2-2/bias*
_output_shapes
:*
dtype0
l

BN_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
BN_1/gamma
e
BN_1/gamma/Read/ReadVariableOpReadVariableOp
BN_1/gamma*
_output_shapes
:*
dtype0
j
	BN_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	BN_1/beta
c
BN_1/beta/Read/ReadVariableOpReadVariableOp	BN_1/beta*
_output_shapes
:*
dtype0
x
BN_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameBN_1/moving_mean
q
$BN_1/moving_mean/Read/ReadVariableOpReadVariableOpBN_1/moving_mean*
_output_shapes
:*
dtype0

BN_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameBN_1/moving_variance
y
(BN_1/moving_variance/Read/ReadVariableOpReadVariableOpBN_1/moving_variance*
_output_shapes
:*
dtype0

Conv_3-1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameConv_3-1/kernel
{
#Conv_3-1/kernel/Read/ReadVariableOpReadVariableOpConv_3-1/kernel*&
_output_shapes
: *
dtype0
r
Conv_3-1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameConv_3-1/bias
k
!Conv_3-1/bias/Read/ReadVariableOpReadVariableOpConv_3-1/bias*
_output_shapes
: *
dtype0

Conv_3-2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  * 
shared_nameConv_3-2/kernel
{
#Conv_3-2/kernel/Read/ReadVariableOpReadVariableOpConv_3-2/kernel*&
_output_shapes
:  *
dtype0
r
Conv_3-2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameConv_3-2/bias
k
!Conv_3-2/bias/Read/ReadVariableOpReadVariableOpConv_3-2/bias*
_output_shapes
: *
dtype0

Conv_4-1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  * 
shared_nameConv_4-1/kernel
{
#Conv_4-1/kernel/Read/ReadVariableOpReadVariableOpConv_4-1/kernel*&
_output_shapes
:  *
dtype0
r
Conv_4-1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameConv_4-1/bias
k
!Conv_4-1/bias/Read/ReadVariableOpReadVariableOpConv_4-1/bias*
_output_shapes
: *
dtype0

Conv_4-2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  * 
shared_nameConv_4-2/kernel
{
#Conv_4-2/kernel/Read/ReadVariableOpReadVariableOpConv_4-2/kernel*&
_output_shapes
:  *
dtype0
r
Conv_4-2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameConv_4-2/bias
k
!Conv_4-2/bias/Read/ReadVariableOpReadVariableOpConv_4-2/bias*
_output_shapes
: *
dtype0
l

BN_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
BN_2/gamma
e
BN_2/gamma/Read/ReadVariableOpReadVariableOp
BN_2/gamma*
_output_shapes
: *
dtype0
j
	BN_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	BN_2/beta
c
BN_2/beta/Read/ReadVariableOpReadVariableOp	BN_2/beta*
_output_shapes
: *
dtype0
x
BN_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameBN_2/moving_mean
q
$BN_2/moving_mean/Read/ReadVariableOpReadVariableOpBN_2/moving_mean*
_output_shapes
: *
dtype0

BN_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameBN_2/moving_variance
y
(BN_2/moving_variance/Read/ReadVariableOpReadVariableOpBN_2/moving_variance*
_output_shapes
: *
dtype0
z
Dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
 *
shared_nameDense_1/kernel
s
"Dense_1/kernel/Read/ReadVariableOpReadVariableOpDense_1/kernel* 
_output_shapes
:
 *
dtype0
q
Dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameDense_1/bias
j
 Dense_1/bias/Read/ReadVariableOpReadVariableOpDense_1/bias*
_output_shapes	
:*
dtype0

Dense_NXP_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*#
shared_nameDense_NXP_2/kernel
{
&Dense_NXP_2/kernel/Read/ReadVariableOpReadVariableOpDense_NXP_2/kernel* 
_output_shapes
:
*
dtype0
y
Dense_NXP_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameDense_NXP_2/bias
r
$Dense_NXP_2/bias/Read/ReadVariableOpReadVariableOpDense_NXP_2/bias*
_output_shapes	
:*
dtype0

Dense_NXP_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*#
shared_nameDense_NXP_3/kernel
{
&Dense_NXP_3/kernel/Read/ReadVariableOpReadVariableOpDense_NXP_3/kernel* 
_output_shapes
:
*
dtype0
y
Dense_NXP_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameDense_NXP_3/bias
r
$Dense_NXP_3/bias/Read/ReadVariableOpReadVariableOpDense_NXP_3/bias*
_output_shapes	
:*
dtype0
y
targets/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_nametargets/kernel
r
"targets/kernel/Read/ReadVariableOpReadVariableOptargets/kernel*
_output_shapes
:	*
dtype0
p
targets/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nametargets/bias
i
 targets/bias/Read/ReadVariableOpReadVariableOptargets/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0

Adam/input/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/input/kernel/m

'Adam/input/kernel/m/Read/ReadVariableOpReadVariableOpAdam/input/kernel/m*&
_output_shapes
:*
dtype0
z
Adam/input/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/input/bias/m
s
%Adam/input/bias/m/Read/ReadVariableOpReadVariableOpAdam/input/bias/m*
_output_shapes
:*
dtype0

Adam/Conv_1-2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/Conv_1-2/kernel/m

*Adam/Conv_1-2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Conv_1-2/kernel/m*&
_output_shapes
:*
dtype0

Adam/Conv_1-2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/Conv_1-2/bias/m
y
(Adam/Conv_1-2/bias/m/Read/ReadVariableOpReadVariableOpAdam/Conv_1-2/bias/m*
_output_shapes
:*
dtype0

Adam/Conv_2-1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/Conv_2-1/kernel/m

*Adam/Conv_2-1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Conv_2-1/kernel/m*&
_output_shapes
:*
dtype0

Adam/Conv_2-1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/Conv_2-1/bias/m
y
(Adam/Conv_2-1/bias/m/Read/ReadVariableOpReadVariableOpAdam/Conv_2-1/bias/m*
_output_shapes
:*
dtype0

Adam/Conv_2-2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/Conv_2-2/kernel/m

*Adam/Conv_2-2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Conv_2-2/kernel/m*&
_output_shapes
:*
dtype0

Adam/Conv_2-2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/Conv_2-2/bias/m
y
(Adam/Conv_2-2/bias/m/Read/ReadVariableOpReadVariableOpAdam/Conv_2-2/bias/m*
_output_shapes
:*
dtype0
z
Adam/BN_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/BN_1/gamma/m
s
%Adam/BN_1/gamma/m/Read/ReadVariableOpReadVariableOpAdam/BN_1/gamma/m*
_output_shapes
:*
dtype0
x
Adam/BN_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameAdam/BN_1/beta/m
q
$Adam/BN_1/beta/m/Read/ReadVariableOpReadVariableOpAdam/BN_1/beta/m*
_output_shapes
:*
dtype0

Adam/Conv_3-1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/Conv_3-1/kernel/m

*Adam/Conv_3-1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Conv_3-1/kernel/m*&
_output_shapes
: *
dtype0

Adam/Conv_3-1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/Conv_3-1/bias/m
y
(Adam/Conv_3-1/bias/m/Read/ReadVariableOpReadVariableOpAdam/Conv_3-1/bias/m*
_output_shapes
: *
dtype0

Adam/Conv_3-2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *'
shared_nameAdam/Conv_3-2/kernel/m

*Adam/Conv_3-2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Conv_3-2/kernel/m*&
_output_shapes
:  *
dtype0

Adam/Conv_3-2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/Conv_3-2/bias/m
y
(Adam/Conv_3-2/bias/m/Read/ReadVariableOpReadVariableOpAdam/Conv_3-2/bias/m*
_output_shapes
: *
dtype0

Adam/Conv_4-1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *'
shared_nameAdam/Conv_4-1/kernel/m

*Adam/Conv_4-1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Conv_4-1/kernel/m*&
_output_shapes
:  *
dtype0

Adam/Conv_4-1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/Conv_4-1/bias/m
y
(Adam/Conv_4-1/bias/m/Read/ReadVariableOpReadVariableOpAdam/Conv_4-1/bias/m*
_output_shapes
: *
dtype0

Adam/Conv_4-2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *'
shared_nameAdam/Conv_4-2/kernel/m

*Adam/Conv_4-2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Conv_4-2/kernel/m*&
_output_shapes
:  *
dtype0

Adam/Conv_4-2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/Conv_4-2/bias/m
y
(Adam/Conv_4-2/bias/m/Read/ReadVariableOpReadVariableOpAdam/Conv_4-2/bias/m*
_output_shapes
: *
dtype0
z
Adam/BN_2/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/BN_2/gamma/m
s
%Adam/BN_2/gamma/m/Read/ReadVariableOpReadVariableOpAdam/BN_2/gamma/m*
_output_shapes
: *
dtype0
x
Adam/BN_2/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameAdam/BN_2/beta/m
q
$Adam/BN_2/beta/m/Read/ReadVariableOpReadVariableOpAdam/BN_2/beta/m*
_output_shapes
: *
dtype0

Adam/Dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
 *&
shared_nameAdam/Dense_1/kernel/m

)Adam/Dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Dense_1/kernel/m* 
_output_shapes
:
 *
dtype0

Adam/Dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/Dense_1/bias/m
x
'Adam/Dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/Dense_1/bias/m*
_output_shapes	
:*
dtype0

Adam/Dense_NXP_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
**
shared_nameAdam/Dense_NXP_2/kernel/m

-Adam/Dense_NXP_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Dense_NXP_2/kernel/m* 
_output_shapes
:
*
dtype0

Adam/Dense_NXP_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/Dense_NXP_2/bias/m

+Adam/Dense_NXP_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/Dense_NXP_2/bias/m*
_output_shapes	
:*
dtype0

Adam/Dense_NXP_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
**
shared_nameAdam/Dense_NXP_3/kernel/m

-Adam/Dense_NXP_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Dense_NXP_3/kernel/m* 
_output_shapes
:
*
dtype0

Adam/Dense_NXP_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/Dense_NXP_3/bias/m

+Adam/Dense_NXP_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/Dense_NXP_3/bias/m*
_output_shapes	
:*
dtype0

Adam/targets/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*&
shared_nameAdam/targets/kernel/m

)Adam/targets/kernel/m/Read/ReadVariableOpReadVariableOpAdam/targets/kernel/m*
_output_shapes
:	*
dtype0
~
Adam/targets/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/targets/bias/m
w
'Adam/targets/bias/m/Read/ReadVariableOpReadVariableOpAdam/targets/bias/m*
_output_shapes
:*
dtype0

Adam/input/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/input/kernel/v

'Adam/input/kernel/v/Read/ReadVariableOpReadVariableOpAdam/input/kernel/v*&
_output_shapes
:*
dtype0
z
Adam/input/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/input/bias/v
s
%Adam/input/bias/v/Read/ReadVariableOpReadVariableOpAdam/input/bias/v*
_output_shapes
:*
dtype0

Adam/Conv_1-2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/Conv_1-2/kernel/v

*Adam/Conv_1-2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Conv_1-2/kernel/v*&
_output_shapes
:*
dtype0

Adam/Conv_1-2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/Conv_1-2/bias/v
y
(Adam/Conv_1-2/bias/v/Read/ReadVariableOpReadVariableOpAdam/Conv_1-2/bias/v*
_output_shapes
:*
dtype0

Adam/Conv_2-1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/Conv_2-1/kernel/v

*Adam/Conv_2-1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Conv_2-1/kernel/v*&
_output_shapes
:*
dtype0

Adam/Conv_2-1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/Conv_2-1/bias/v
y
(Adam/Conv_2-1/bias/v/Read/ReadVariableOpReadVariableOpAdam/Conv_2-1/bias/v*
_output_shapes
:*
dtype0

Adam/Conv_2-2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/Conv_2-2/kernel/v

*Adam/Conv_2-2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Conv_2-2/kernel/v*&
_output_shapes
:*
dtype0

Adam/Conv_2-2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/Conv_2-2/bias/v
y
(Adam/Conv_2-2/bias/v/Read/ReadVariableOpReadVariableOpAdam/Conv_2-2/bias/v*
_output_shapes
:*
dtype0
z
Adam/BN_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/BN_1/gamma/v
s
%Adam/BN_1/gamma/v/Read/ReadVariableOpReadVariableOpAdam/BN_1/gamma/v*
_output_shapes
:*
dtype0
x
Adam/BN_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameAdam/BN_1/beta/v
q
$Adam/BN_1/beta/v/Read/ReadVariableOpReadVariableOpAdam/BN_1/beta/v*
_output_shapes
:*
dtype0

Adam/Conv_3-1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/Conv_3-1/kernel/v

*Adam/Conv_3-1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Conv_3-1/kernel/v*&
_output_shapes
: *
dtype0

Adam/Conv_3-1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/Conv_3-1/bias/v
y
(Adam/Conv_3-1/bias/v/Read/ReadVariableOpReadVariableOpAdam/Conv_3-1/bias/v*
_output_shapes
: *
dtype0

Adam/Conv_3-2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *'
shared_nameAdam/Conv_3-2/kernel/v

*Adam/Conv_3-2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Conv_3-2/kernel/v*&
_output_shapes
:  *
dtype0

Adam/Conv_3-2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/Conv_3-2/bias/v
y
(Adam/Conv_3-2/bias/v/Read/ReadVariableOpReadVariableOpAdam/Conv_3-2/bias/v*
_output_shapes
: *
dtype0

Adam/Conv_4-1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *'
shared_nameAdam/Conv_4-1/kernel/v

*Adam/Conv_4-1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Conv_4-1/kernel/v*&
_output_shapes
:  *
dtype0

Adam/Conv_4-1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/Conv_4-1/bias/v
y
(Adam/Conv_4-1/bias/v/Read/ReadVariableOpReadVariableOpAdam/Conv_4-1/bias/v*
_output_shapes
: *
dtype0

Adam/Conv_4-2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *'
shared_nameAdam/Conv_4-2/kernel/v

*Adam/Conv_4-2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Conv_4-2/kernel/v*&
_output_shapes
:  *
dtype0

Adam/Conv_4-2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/Conv_4-2/bias/v
y
(Adam/Conv_4-2/bias/v/Read/ReadVariableOpReadVariableOpAdam/Conv_4-2/bias/v*
_output_shapes
: *
dtype0
z
Adam/BN_2/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/BN_2/gamma/v
s
%Adam/BN_2/gamma/v/Read/ReadVariableOpReadVariableOpAdam/BN_2/gamma/v*
_output_shapes
: *
dtype0
x
Adam/BN_2/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameAdam/BN_2/beta/v
q
$Adam/BN_2/beta/v/Read/ReadVariableOpReadVariableOpAdam/BN_2/beta/v*
_output_shapes
: *
dtype0

Adam/Dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
 *&
shared_nameAdam/Dense_1/kernel/v

)Adam/Dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Dense_1/kernel/v* 
_output_shapes
:
 *
dtype0

Adam/Dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/Dense_1/bias/v
x
'Adam/Dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/Dense_1/bias/v*
_output_shapes	
:*
dtype0

Adam/Dense_NXP_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
**
shared_nameAdam/Dense_NXP_2/kernel/v

-Adam/Dense_NXP_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Dense_NXP_2/kernel/v* 
_output_shapes
:
*
dtype0

Adam/Dense_NXP_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/Dense_NXP_2/bias/v

+Adam/Dense_NXP_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/Dense_NXP_2/bias/v*
_output_shapes	
:*
dtype0

Adam/Dense_NXP_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
**
shared_nameAdam/Dense_NXP_3/kernel/v

-Adam/Dense_NXP_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Dense_NXP_3/kernel/v* 
_output_shapes
:
*
dtype0

Adam/Dense_NXP_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/Dense_NXP_3/bias/v

+Adam/Dense_NXP_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/Dense_NXP_3/bias/v*
_output_shapes	
:*
dtype0

Adam/targets/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*&
shared_nameAdam/targets/kernel/v

)Adam/targets/kernel/v/Read/ReadVariableOpReadVariableOpAdam/targets/kernel/v*
_output_shapes
:	*
dtype0
~
Adam/targets/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/targets/bias/v
w
'Adam/targets/bias/v/Read/ReadVariableOpReadVariableOpAdam/targets/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
Љ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*РЈ
valueЕЈBБЈ BЉЈ
А
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer-6
layer_with_weights-5
layer-7
	layer_with_weights-6
	layer-8

layer-9
layer_with_weights-7
layer-10
layer_with_weights-8
layer-11
layer_with_weights-9
layer-12
layer-13
layer-14
layer_with_weights-10
layer-15
layer-16
layer_with_weights-11
layer-17
layer-18
layer_with_weights-12
layer-19
layer-20
layer_with_weights-13
layer-21
	optimizer

signatures
#_self_saveable_object_factories
regularization_losses
trainable_variables
	variables
	keras_api


kernel
bias
# _self_saveable_object_factories
!regularization_losses
"trainable_variables
#	variables
$	keras_api


%kernel
&bias
#'_self_saveable_object_factories
(regularization_losses
)trainable_variables
*	variables
+	keras_api
w
#,_self_saveable_object_factories
-regularization_losses
.trainable_variables
/	variables
0	keras_api


1kernel
2bias
#3_self_saveable_object_factories
4regularization_losses
5trainable_variables
6	variables
7	keras_api


8kernel
9bias
#:_self_saveable_object_factories
;regularization_losses
<trainable_variables
=	variables
>	keras_api
М
?axis
	@gamma
Abeta
Bmoving_mean
Cmoving_variance
#D_self_saveable_object_factories
Eregularization_losses
Ftrainable_variables
G	variables
H	keras_api
w
#I_self_saveable_object_factories
Jregularization_losses
Ktrainable_variables
L	variables
M	keras_api


Nkernel
Obias
#P_self_saveable_object_factories
Qregularization_losses
Rtrainable_variables
S	variables
T	keras_api


Ukernel
Vbias
#W_self_saveable_object_factories
Xregularization_losses
Ytrainable_variables
Z	variables
[	keras_api
w
#\_self_saveable_object_factories
]regularization_losses
^trainable_variables
_	variables
`	keras_api


akernel
bbias
#c_self_saveable_object_factories
dregularization_losses
etrainable_variables
f	variables
g	keras_api


hkernel
ibias
#j_self_saveable_object_factories
kregularization_losses
ltrainable_variables
m	variables
n	keras_api
М
oaxis
	pgamma
qbeta
rmoving_mean
smoving_variance
#t_self_saveable_object_factories
uregularization_losses
vtrainable_variables
w	variables
x	keras_api
w
#y_self_saveable_object_factories
zregularization_losses
{trainable_variables
|	variables
}	keras_api
z
#~_self_saveable_object_factories
regularization_losses
trainable_variables
	variables
	keras_api

kernel
	bias
$_self_saveable_object_factories
regularization_losses
trainable_variables
	variables
	keras_api
|
$_self_saveable_object_factories
regularization_losses
trainable_variables
	variables
	keras_api

kernel
	bias
$_self_saveable_object_factories
regularization_losses
trainable_variables
	variables
	keras_api
|
$_self_saveable_object_factories
regularization_losses
trainable_variables
	variables
	keras_api

kernel
	bias
$_self_saveable_object_factories
regularization_losses
trainable_variables
 	variables
Ё	keras_api
|
$Ђ_self_saveable_object_factories
Ѓregularization_losses
Єtrainable_variables
Ѕ	variables
І	keras_api

Їkernel
	Јbias
$Љ_self_saveable_object_factories
Њregularization_losses
Ћtrainable_variables
Ќ	variables
­	keras_api

	Ўiter
Џbeta_1
Аbeta_2

Бdecay
Вlearning_ratemЖmЗ%mИ&mЙ1mК2mЛ8mМ9mН@mОAmПNmРOmСUmТVmУamФbmХhmЦimЧpmШqmЩ	mЪ	mЫ	mЬ	mЭ	mЮ	mЯ	Їmа	Јmбvвvг%vд&vе1vж2vз8vи9vй@vкAvлNvмOvнUvоVvпavрbvсhvтivуpvфqvх	vц	vч	vш	vщ	vъ	vы	Їvь	Јvэ
 
 
 
о
0
1
%2
&3
14
25
86
97
@8
A9
N10
O11
U12
V13
a14
b15
h16
i17
p18
q19
20
21
22
23
24
25
Ї26
Ј27
ў
0
1
%2
&3
14
25
86
97
@8
A9
B10
C11
N12
O13
U14
V15
a16
b17
h18
i19
p20
q21
r22
s23
24
25
26
27
28
29
Ї30
Ј31
В
Гlayers
regularization_losses
trainable_variables
 Дlayer_regularization_losses
Еlayer_metrics
Жmetrics
	variables
Зnon_trainable_variables
XV
VARIABLE_VALUEinput/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
input/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1

0
1
В
Иlayers
!regularization_losses
"trainable_variables
 Йlayer_regularization_losses
Кlayer_metrics
Лmetrics
#	variables
Мnon_trainable_variables
[Y
VARIABLE_VALUEConv_1-2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEConv_1-2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

%0
&1

%0
&1
В
Нlayers
(regularization_losses
)trainable_variables
 Оlayer_regularization_losses
Пlayer_metrics
Рmetrics
*	variables
Сnon_trainable_variables
 
 
 
 
В
Тlayers
-regularization_losses
.trainable_variables
 Уlayer_regularization_losses
Фlayer_metrics
Хmetrics
/	variables
Цnon_trainable_variables
[Y
VARIABLE_VALUEConv_2-1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEConv_2-1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

10
21

10
21
В
Чlayers
4regularization_losses
5trainable_variables
 Шlayer_regularization_losses
Щlayer_metrics
Ъmetrics
6	variables
Ыnon_trainable_variables
[Y
VARIABLE_VALUEConv_2-2/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEConv_2-2/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

80
91

80
91
В
Ьlayers
;regularization_losses
<trainable_variables
 Эlayer_regularization_losses
Юlayer_metrics
Яmetrics
=	variables
аnon_trainable_variables
 
US
VARIABLE_VALUE
BN_1/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	BN_1/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEBN_1/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUEBN_1/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
 

@0
A1

@0
A1
B2
C3
В
бlayers
Eregularization_losses
Ftrainable_variables
 вlayer_regularization_losses
гlayer_metrics
дmetrics
G	variables
еnon_trainable_variables
 
 
 
 
В
жlayers
Jregularization_losses
Ktrainable_variables
 зlayer_regularization_losses
иlayer_metrics
йmetrics
L	variables
кnon_trainable_variables
[Y
VARIABLE_VALUEConv_3-1/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEConv_3-1/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

N0
O1

N0
O1
В
лlayers
Qregularization_losses
Rtrainable_variables
 мlayer_regularization_losses
нlayer_metrics
оmetrics
S	variables
пnon_trainable_variables
[Y
VARIABLE_VALUEConv_3-2/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEConv_3-2/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

U0
V1

U0
V1
В
рlayers
Xregularization_losses
Ytrainable_variables
 сlayer_regularization_losses
тlayer_metrics
уmetrics
Z	variables
фnon_trainable_variables
 
 
 
 
В
хlayers
]regularization_losses
^trainable_variables
 цlayer_regularization_losses
чlayer_metrics
шmetrics
_	variables
щnon_trainable_variables
[Y
VARIABLE_VALUEConv_4-1/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEConv_4-1/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

a0
b1

a0
b1
В
ъlayers
dregularization_losses
etrainable_variables
 ыlayer_regularization_losses
ьlayer_metrics
эmetrics
f	variables
юnon_trainable_variables
[Y
VARIABLE_VALUEConv_4-2/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEConv_4-2/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

h0
i1

h0
i1
В
яlayers
kregularization_losses
ltrainable_variables
 №layer_regularization_losses
ёlayer_metrics
ђmetrics
m	variables
ѓnon_trainable_variables
 
US
VARIABLE_VALUE
BN_2/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	BN_2/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEBN_2/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUEBN_2/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
 

p0
q1

p0
q1
r2
s3
В
єlayers
uregularization_losses
vtrainable_variables
 ѕlayer_regularization_losses
іlayer_metrics
їmetrics
w	variables
јnon_trainable_variables
 
 
 
 
В
љlayers
zregularization_losses
{trainable_variables
 њlayer_regularization_losses
ћlayer_metrics
ќmetrics
|	variables
§non_trainable_variables
 
 
 
 
Д
ўlayers
regularization_losses
trainable_variables
 џlayer_regularization_losses
layer_metrics
metrics
	variables
non_trainable_variables
[Y
VARIABLE_VALUEDense_1/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEDense_1/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1

0
1
Е
layers
regularization_losses
trainable_variables
 layer_regularization_losses
layer_metrics
metrics
	variables
non_trainable_variables
 
 
 
 
Е
layers
regularization_losses
trainable_variables
 layer_regularization_losses
layer_metrics
metrics
	variables
non_trainable_variables
_]
VARIABLE_VALUEDense_NXP_2/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEDense_NXP_2/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1

0
1
Е
layers
regularization_losses
trainable_variables
 layer_regularization_losses
layer_metrics
metrics
	variables
non_trainable_variables
 
 
 
 
Е
layers
regularization_losses
trainable_variables
 layer_regularization_losses
layer_metrics
metrics
	variables
non_trainable_variables
_]
VARIABLE_VALUEDense_NXP_3/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEDense_NXP_3/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1

0
1
Е
layers
regularization_losses
trainable_variables
 layer_regularization_losses
layer_metrics
metrics
 	variables
non_trainable_variables
 
 
 
 
Е
layers
Ѓregularization_losses
Єtrainable_variables
 layer_regularization_losses
layer_metrics
metrics
Ѕ	variables
 non_trainable_variables
[Y
VARIABLE_VALUEtargets/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEtargets/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

Ї0
Ј1

Ї0
Ј1
Е
Ёlayers
Њregularization_losses
Ћtrainable_variables
 Ђlayer_regularization_losses
Ѓlayer_metrics
Єmetrics
Ќ	variables
Ѕnon_trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
І
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
 
 

І0
Ї1
Ј2

B0
C1
r2
s3
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

B0
C1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

r0
s1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

Љtotal

Њcount
Ћ	variables
Ќ	keras_api
I

­total

Ўcount
Џ
_fn_kwargs
А	variables
Б	keras_api
8

Вtotal

Гcount
Д	variables
Е	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

Љ0
Њ1

Ћ	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

­0
Ў1

А	variables
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE

В0
Г1

Д	variables
{y
VARIABLE_VALUEAdam/input/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/input/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv_1-2/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Conv_1-2/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv_2-1/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Conv_2-1/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv_2-2/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Conv_2-2/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/BN_1/gamma/mQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/BN_1/beta/mPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv_3-1/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Conv_3-1/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv_3-2/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Conv_3-2/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv_4-1/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Conv_4-1/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv_4-2/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Conv_4-2/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/BN_2/gamma/mQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/BN_2/beta/mPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Dense_1/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Dense_1/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/Dense_NXP_2/kernel/mSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Dense_NXP_2/bias/mQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/Dense_NXP_3/kernel/mSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Dense_NXP_3/bias/mQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/targets/kernel/mSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/targets/bias/mQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/input/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/input/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv_1-2/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Conv_1-2/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv_2-1/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Conv_2-1/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv_2-2/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Conv_2-2/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/BN_1/gamma/vQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/BN_1/beta/vPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv_3-1/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Conv_3-1/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv_3-2/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Conv_3-2/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv_4-1/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Conv_4-1/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv_4-2/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Conv_4-2/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/BN_2/gamma/vQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/BN_2/beta/vPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Dense_1/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Dense_1/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/Dense_NXP_2/kernel/vSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Dense_NXP_2/bias/vQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/Dense_NXP_3/kernel/vSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Dense_NXP_3/bias/vQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/targets/kernel/vSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/targets/bias/vQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_input_inputPlaceholder*1
_output_shapes
:џџџџџџџџџии*
dtype0*&
shape:џџџџџџџџџии
ј
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_inputinput/kernel
input/biasConv_1-2/kernelConv_1-2/biasConv_2-1/kernelConv_2-1/biasConv_2-2/kernelConv_2-2/bias
BN_1/gamma	BN_1/betaBN_1/moving_meanBN_1/moving_varianceConv_3-1/kernelConv_3-1/biasConv_3-2/kernelConv_3-2/biasConv_4-1/kernelConv_4-1/biasConv_4-2/kernelConv_4-2/bias
BN_2/gamma	BN_2/betaBN_2/moving_meanBN_2/moving_varianceDense_1/kernelDense_1/biasDense_NXP_2/kernelDense_NXP_2/biasDense_NXP_3/kernelDense_NXP_3/biastargets/kerneltargets/bias*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*B
_read_only_resource_inputs$
" 	
 *0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference_signature_wrapper_34257
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
!
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename input/kernel/Read/ReadVariableOpinput/bias/Read/ReadVariableOp#Conv_1-2/kernel/Read/ReadVariableOp!Conv_1-2/bias/Read/ReadVariableOp#Conv_2-1/kernel/Read/ReadVariableOp!Conv_2-1/bias/Read/ReadVariableOp#Conv_2-2/kernel/Read/ReadVariableOp!Conv_2-2/bias/Read/ReadVariableOpBN_1/gamma/Read/ReadVariableOpBN_1/beta/Read/ReadVariableOp$BN_1/moving_mean/Read/ReadVariableOp(BN_1/moving_variance/Read/ReadVariableOp#Conv_3-1/kernel/Read/ReadVariableOp!Conv_3-1/bias/Read/ReadVariableOp#Conv_3-2/kernel/Read/ReadVariableOp!Conv_3-2/bias/Read/ReadVariableOp#Conv_4-1/kernel/Read/ReadVariableOp!Conv_4-1/bias/Read/ReadVariableOp#Conv_4-2/kernel/Read/ReadVariableOp!Conv_4-2/bias/Read/ReadVariableOpBN_2/gamma/Read/ReadVariableOpBN_2/beta/Read/ReadVariableOp$BN_2/moving_mean/Read/ReadVariableOp(BN_2/moving_variance/Read/ReadVariableOp"Dense_1/kernel/Read/ReadVariableOp Dense_1/bias/Read/ReadVariableOp&Dense_NXP_2/kernel/Read/ReadVariableOp$Dense_NXP_2/bias/Read/ReadVariableOp&Dense_NXP_3/kernel/Read/ReadVariableOp$Dense_NXP_3/bias/Read/ReadVariableOp"targets/kernel/Read/ReadVariableOp targets/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOp'Adam/input/kernel/m/Read/ReadVariableOp%Adam/input/bias/m/Read/ReadVariableOp*Adam/Conv_1-2/kernel/m/Read/ReadVariableOp(Adam/Conv_1-2/bias/m/Read/ReadVariableOp*Adam/Conv_2-1/kernel/m/Read/ReadVariableOp(Adam/Conv_2-1/bias/m/Read/ReadVariableOp*Adam/Conv_2-2/kernel/m/Read/ReadVariableOp(Adam/Conv_2-2/bias/m/Read/ReadVariableOp%Adam/BN_1/gamma/m/Read/ReadVariableOp$Adam/BN_1/beta/m/Read/ReadVariableOp*Adam/Conv_3-1/kernel/m/Read/ReadVariableOp(Adam/Conv_3-1/bias/m/Read/ReadVariableOp*Adam/Conv_3-2/kernel/m/Read/ReadVariableOp(Adam/Conv_3-2/bias/m/Read/ReadVariableOp*Adam/Conv_4-1/kernel/m/Read/ReadVariableOp(Adam/Conv_4-1/bias/m/Read/ReadVariableOp*Adam/Conv_4-2/kernel/m/Read/ReadVariableOp(Adam/Conv_4-2/bias/m/Read/ReadVariableOp%Adam/BN_2/gamma/m/Read/ReadVariableOp$Adam/BN_2/beta/m/Read/ReadVariableOp)Adam/Dense_1/kernel/m/Read/ReadVariableOp'Adam/Dense_1/bias/m/Read/ReadVariableOp-Adam/Dense_NXP_2/kernel/m/Read/ReadVariableOp+Adam/Dense_NXP_2/bias/m/Read/ReadVariableOp-Adam/Dense_NXP_3/kernel/m/Read/ReadVariableOp+Adam/Dense_NXP_3/bias/m/Read/ReadVariableOp)Adam/targets/kernel/m/Read/ReadVariableOp'Adam/targets/bias/m/Read/ReadVariableOp'Adam/input/kernel/v/Read/ReadVariableOp%Adam/input/bias/v/Read/ReadVariableOp*Adam/Conv_1-2/kernel/v/Read/ReadVariableOp(Adam/Conv_1-2/bias/v/Read/ReadVariableOp*Adam/Conv_2-1/kernel/v/Read/ReadVariableOp(Adam/Conv_2-1/bias/v/Read/ReadVariableOp*Adam/Conv_2-2/kernel/v/Read/ReadVariableOp(Adam/Conv_2-2/bias/v/Read/ReadVariableOp%Adam/BN_1/gamma/v/Read/ReadVariableOp$Adam/BN_1/beta/v/Read/ReadVariableOp*Adam/Conv_3-1/kernel/v/Read/ReadVariableOp(Adam/Conv_3-1/bias/v/Read/ReadVariableOp*Adam/Conv_3-2/kernel/v/Read/ReadVariableOp(Adam/Conv_3-2/bias/v/Read/ReadVariableOp*Adam/Conv_4-1/kernel/v/Read/ReadVariableOp(Adam/Conv_4-1/bias/v/Read/ReadVariableOp*Adam/Conv_4-2/kernel/v/Read/ReadVariableOp(Adam/Conv_4-2/bias/v/Read/ReadVariableOp%Adam/BN_2/gamma/v/Read/ReadVariableOp$Adam/BN_2/beta/v/Read/ReadVariableOp)Adam/Dense_1/kernel/v/Read/ReadVariableOp'Adam/Dense_1/bias/v/Read/ReadVariableOp-Adam/Dense_NXP_2/kernel/v/Read/ReadVariableOp+Adam/Dense_NXP_2/bias/v/Read/ReadVariableOp-Adam/Dense_NXP_3/kernel/v/Read/ReadVariableOp+Adam/Dense_NXP_3/bias/v/Read/ReadVariableOp)Adam/targets/kernel/v/Read/ReadVariableOp'Adam/targets/bias/v/Read/ReadVariableOpConst*p
Tini
g2e	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *'
f"R 
__inference__traced_save_35566
к
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameinput/kernel
input/biasConv_1-2/kernelConv_1-2/biasConv_2-1/kernelConv_2-1/biasConv_2-2/kernelConv_2-2/bias
BN_1/gamma	BN_1/betaBN_1/moving_meanBN_1/moving_varianceConv_3-1/kernelConv_3-1/biasConv_3-2/kernelConv_3-2/biasConv_4-1/kernelConv_4-1/biasConv_4-2/kernelConv_4-2/bias
BN_2/gamma	BN_2/betaBN_2/moving_meanBN_2/moving_varianceDense_1/kernelDense_1/biasDense_NXP_2/kernelDense_NXP_2/biasDense_NXP_3/kernelDense_NXP_3/biastargets/kerneltargets/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1total_2count_2Adam/input/kernel/mAdam/input/bias/mAdam/Conv_1-2/kernel/mAdam/Conv_1-2/bias/mAdam/Conv_2-1/kernel/mAdam/Conv_2-1/bias/mAdam/Conv_2-2/kernel/mAdam/Conv_2-2/bias/mAdam/BN_1/gamma/mAdam/BN_1/beta/mAdam/Conv_3-1/kernel/mAdam/Conv_3-1/bias/mAdam/Conv_3-2/kernel/mAdam/Conv_3-2/bias/mAdam/Conv_4-1/kernel/mAdam/Conv_4-1/bias/mAdam/Conv_4-2/kernel/mAdam/Conv_4-2/bias/mAdam/BN_2/gamma/mAdam/BN_2/beta/mAdam/Dense_1/kernel/mAdam/Dense_1/bias/mAdam/Dense_NXP_2/kernel/mAdam/Dense_NXP_2/bias/mAdam/Dense_NXP_3/kernel/mAdam/Dense_NXP_3/bias/mAdam/targets/kernel/mAdam/targets/bias/mAdam/input/kernel/vAdam/input/bias/vAdam/Conv_1-2/kernel/vAdam/Conv_1-2/bias/vAdam/Conv_2-1/kernel/vAdam/Conv_2-1/bias/vAdam/Conv_2-2/kernel/vAdam/Conv_2-2/bias/vAdam/BN_1/gamma/vAdam/BN_1/beta/vAdam/Conv_3-1/kernel/vAdam/Conv_3-1/bias/vAdam/Conv_3-2/kernel/vAdam/Conv_3-2/bias/vAdam/Conv_4-1/kernel/vAdam/Conv_4-1/bias/vAdam/Conv_4-2/kernel/vAdam/Conv_4-2/bias/vAdam/BN_2/gamma/vAdam/BN_2/beta/vAdam/Dense_1/kernel/vAdam/Dense_1/bias/vAdam/Dense_NXP_2/kernel/vAdam/Dense_NXP_2/bias/vAdam/Dense_NXP_3/kernel/vAdam/Dense_NXP_3/bias/vAdam/targets/kernel/vAdam/targets/bias/v*o
Tinh
f2d*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__traced_restore_35873њ
Ш
П
$__inference_BN_2_layer_call_fn_34963

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identityЂStatefulPartitionedCallІ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_BN_2_layer_call_and_return_conditional_losses_329382
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs
К

њ
F__inference_Dense_NXP_2_layer_call_and_return_conditional_losses_33298

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ц

%__inference_input_layer_call_fn_34675

inputs!
unknown:
	unknown_0:
identityЂStatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџжж*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_input_layer_call_and_return_conditional_losses_330722
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:џџџџџџџџџжж2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџии: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:џџџџџџџџџии
 
_user_specified_nameinputs
ЭЃ
њ<
!__inference__traced_restore_35873
file_prefix7
assignvariableop_input_kernel:+
assignvariableop_1_input_bias:<
"assignvariableop_2_conv_1_2_kernel:.
 assignvariableop_3_conv_1_2_bias:<
"assignvariableop_4_conv_2_1_kernel:.
 assignvariableop_5_conv_2_1_bias:<
"assignvariableop_6_conv_2_2_kernel:.
 assignvariableop_7_conv_2_2_bias:+
assignvariableop_8_bn_1_gamma:*
assignvariableop_9_bn_1_beta:2
$assignvariableop_10_bn_1_moving_mean:6
(assignvariableop_11_bn_1_moving_variance:=
#assignvariableop_12_conv_3_1_kernel: /
!assignvariableop_13_conv_3_1_bias: =
#assignvariableop_14_conv_3_2_kernel:  /
!assignvariableop_15_conv_3_2_bias: =
#assignvariableop_16_conv_4_1_kernel:  /
!assignvariableop_17_conv_4_1_bias: =
#assignvariableop_18_conv_4_2_kernel:  /
!assignvariableop_19_conv_4_2_bias: ,
assignvariableop_20_bn_2_gamma: +
assignvariableop_21_bn_2_beta: 2
$assignvariableop_22_bn_2_moving_mean: 6
(assignvariableop_23_bn_2_moving_variance: 6
"assignvariableop_24_dense_1_kernel:
 /
 assignvariableop_25_dense_1_bias:	:
&assignvariableop_26_dense_nxp_2_kernel:
3
$assignvariableop_27_dense_nxp_2_bias:	:
&assignvariableop_28_dense_nxp_3_kernel:
3
$assignvariableop_29_dense_nxp_3_bias:	5
"assignvariableop_30_targets_kernel:	.
 assignvariableop_31_targets_bias:'
assignvariableop_32_adam_iter:	 )
assignvariableop_33_adam_beta_1: )
assignvariableop_34_adam_beta_2: (
assignvariableop_35_adam_decay: 0
&assignvariableop_36_adam_learning_rate: #
assignvariableop_37_total: #
assignvariableop_38_count: %
assignvariableop_39_total_1: %
assignvariableop_40_count_1: %
assignvariableop_41_total_2: %
assignvariableop_42_count_2: A
'assignvariableop_43_adam_input_kernel_m:3
%assignvariableop_44_adam_input_bias_m:D
*assignvariableop_45_adam_conv_1_2_kernel_m:6
(assignvariableop_46_adam_conv_1_2_bias_m:D
*assignvariableop_47_adam_conv_2_1_kernel_m:6
(assignvariableop_48_adam_conv_2_1_bias_m:D
*assignvariableop_49_adam_conv_2_2_kernel_m:6
(assignvariableop_50_adam_conv_2_2_bias_m:3
%assignvariableop_51_adam_bn_1_gamma_m:2
$assignvariableop_52_adam_bn_1_beta_m:D
*assignvariableop_53_adam_conv_3_1_kernel_m: 6
(assignvariableop_54_adam_conv_3_1_bias_m: D
*assignvariableop_55_adam_conv_3_2_kernel_m:  6
(assignvariableop_56_adam_conv_3_2_bias_m: D
*assignvariableop_57_adam_conv_4_1_kernel_m:  6
(assignvariableop_58_adam_conv_4_1_bias_m: D
*assignvariableop_59_adam_conv_4_2_kernel_m:  6
(assignvariableop_60_adam_conv_4_2_bias_m: 3
%assignvariableop_61_adam_bn_2_gamma_m: 2
$assignvariableop_62_adam_bn_2_beta_m: =
)assignvariableop_63_adam_dense_1_kernel_m:
 6
'assignvariableop_64_adam_dense_1_bias_m:	A
-assignvariableop_65_adam_dense_nxp_2_kernel_m:
:
+assignvariableop_66_adam_dense_nxp_2_bias_m:	A
-assignvariableop_67_adam_dense_nxp_3_kernel_m:
:
+assignvariableop_68_adam_dense_nxp_3_bias_m:	<
)assignvariableop_69_adam_targets_kernel_m:	5
'assignvariableop_70_adam_targets_bias_m:A
'assignvariableop_71_adam_input_kernel_v:3
%assignvariableop_72_adam_input_bias_v:D
*assignvariableop_73_adam_conv_1_2_kernel_v:6
(assignvariableop_74_adam_conv_1_2_bias_v:D
*assignvariableop_75_adam_conv_2_1_kernel_v:6
(assignvariableop_76_adam_conv_2_1_bias_v:D
*assignvariableop_77_adam_conv_2_2_kernel_v:6
(assignvariableop_78_adam_conv_2_2_bias_v:3
%assignvariableop_79_adam_bn_1_gamma_v:2
$assignvariableop_80_adam_bn_1_beta_v:D
*assignvariableop_81_adam_conv_3_1_kernel_v: 6
(assignvariableop_82_adam_conv_3_1_bias_v: D
*assignvariableop_83_adam_conv_3_2_kernel_v:  6
(assignvariableop_84_adam_conv_3_2_bias_v: D
*assignvariableop_85_adam_conv_4_1_kernel_v:  6
(assignvariableop_86_adam_conv_4_1_bias_v: D
*assignvariableop_87_adam_conv_4_2_kernel_v:  6
(assignvariableop_88_adam_conv_4_2_bias_v: 3
%assignvariableop_89_adam_bn_2_gamma_v: 2
$assignvariableop_90_adam_bn_2_beta_v: =
)assignvariableop_91_adam_dense_1_kernel_v:
 6
'assignvariableop_92_adam_dense_1_bias_v:	A
-assignvariableop_93_adam_dense_nxp_2_kernel_v:
:
+assignvariableop_94_adam_dense_nxp_2_bias_v:	A
-assignvariableop_95_adam_dense_nxp_3_kernel_v:
:
+assignvariableop_96_adam_dense_nxp_3_bias_v:	<
)assignvariableop_97_adam_targets_kernel_v:	5
'assignvariableop_98_adam_targets_bias_v:
identity_100ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_30ЂAssignVariableOp_31ЂAssignVariableOp_32ЂAssignVariableOp_33ЂAssignVariableOp_34ЂAssignVariableOp_35ЂAssignVariableOp_36ЂAssignVariableOp_37ЂAssignVariableOp_38ЂAssignVariableOp_39ЂAssignVariableOp_4ЂAssignVariableOp_40ЂAssignVariableOp_41ЂAssignVariableOp_42ЂAssignVariableOp_43ЂAssignVariableOp_44ЂAssignVariableOp_45ЂAssignVariableOp_46ЂAssignVariableOp_47ЂAssignVariableOp_48ЂAssignVariableOp_49ЂAssignVariableOp_5ЂAssignVariableOp_50ЂAssignVariableOp_51ЂAssignVariableOp_52ЂAssignVariableOp_53ЂAssignVariableOp_54ЂAssignVariableOp_55ЂAssignVariableOp_56ЂAssignVariableOp_57ЂAssignVariableOp_58ЂAssignVariableOp_59ЂAssignVariableOp_6ЂAssignVariableOp_60ЂAssignVariableOp_61ЂAssignVariableOp_62ЂAssignVariableOp_63ЂAssignVariableOp_64ЂAssignVariableOp_65ЂAssignVariableOp_66ЂAssignVariableOp_67ЂAssignVariableOp_68ЂAssignVariableOp_69ЂAssignVariableOp_7ЂAssignVariableOp_70ЂAssignVariableOp_71ЂAssignVariableOp_72ЂAssignVariableOp_73ЂAssignVariableOp_74ЂAssignVariableOp_75ЂAssignVariableOp_76ЂAssignVariableOp_77ЂAssignVariableOp_78ЂAssignVariableOp_79ЂAssignVariableOp_8ЂAssignVariableOp_80ЂAssignVariableOp_81ЂAssignVariableOp_82ЂAssignVariableOp_83ЂAssignVariableOp_84ЂAssignVariableOp_85ЂAssignVariableOp_86ЂAssignVariableOp_87ЂAssignVariableOp_88ЂAssignVariableOp_89ЂAssignVariableOp_9ЂAssignVariableOp_90ЂAssignVariableOp_91ЂAssignVariableOp_92ЂAssignVariableOp_93ЂAssignVariableOp_94ЂAssignVariableOp_95ЂAssignVariableOp_96ЂAssignVariableOp_97ЂAssignVariableOp_988
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:d*
dtype0*7
value7B7dB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesй
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:d*
dtype0*н
valueгBаdB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesЂ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*І
_output_shapes
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*r
dtypesh
f2d	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_input_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Ђ
AssignVariableOp_1AssignVariableOpassignvariableop_1_input_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Ї
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv_1_2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Ѕ
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv_1_2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4Ї
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv_2_1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5Ѕ
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv_2_1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6Ї
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv_2_2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Ѕ
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv_2_2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8Ђ
AssignVariableOp_8AssignVariableOpassignvariableop_8_bn_1_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Ё
AssignVariableOp_9AssignVariableOpassignvariableop_9_bn_1_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Ќ
AssignVariableOp_10AssignVariableOp$assignvariableop_10_bn_1_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11А
AssignVariableOp_11AssignVariableOp(assignvariableop_11_bn_1_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Ћ
AssignVariableOp_12AssignVariableOp#assignvariableop_12_conv_3_1_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Љ
AssignVariableOp_13AssignVariableOp!assignvariableop_13_conv_3_1_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Ћ
AssignVariableOp_14AssignVariableOp#assignvariableop_14_conv_3_2_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15Љ
AssignVariableOp_15AssignVariableOp!assignvariableop_15_conv_3_2_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Ћ
AssignVariableOp_16AssignVariableOp#assignvariableop_16_conv_4_1_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17Љ
AssignVariableOp_17AssignVariableOp!assignvariableop_17_conv_4_1_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18Ћ
AssignVariableOp_18AssignVariableOp#assignvariableop_18_conv_4_2_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Љ
AssignVariableOp_19AssignVariableOp!assignvariableop_19_conv_4_2_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20І
AssignVariableOp_20AssignVariableOpassignvariableop_20_bn_2_gammaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21Ѕ
AssignVariableOp_21AssignVariableOpassignvariableop_21_bn_2_betaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22Ќ
AssignVariableOp_22AssignVariableOp$assignvariableop_22_bn_2_moving_meanIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23А
AssignVariableOp_23AssignVariableOp(assignvariableop_23_bn_2_moving_varianceIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24Њ
AssignVariableOp_24AssignVariableOp"assignvariableop_24_dense_1_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25Ј
AssignVariableOp_25AssignVariableOp assignvariableop_25_dense_1_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26Ў
AssignVariableOp_26AssignVariableOp&assignvariableop_26_dense_nxp_2_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27Ќ
AssignVariableOp_27AssignVariableOp$assignvariableop_27_dense_nxp_2_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28Ў
AssignVariableOp_28AssignVariableOp&assignvariableop_28_dense_nxp_3_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29Ќ
AssignVariableOp_29AssignVariableOp$assignvariableop_29_dense_nxp_3_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30Њ
AssignVariableOp_30AssignVariableOp"assignvariableop_30_targets_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31Ј
AssignVariableOp_31AssignVariableOp assignvariableop_31_targets_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_32Ѕ
AssignVariableOp_32AssignVariableOpassignvariableop_32_adam_iterIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33Ї
AssignVariableOp_33AssignVariableOpassignvariableop_33_adam_beta_1Identity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34Ї
AssignVariableOp_34AssignVariableOpassignvariableop_34_adam_beta_2Identity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35І
AssignVariableOp_35AssignVariableOpassignvariableop_35_adam_decayIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36Ў
AssignVariableOp_36AssignVariableOp&assignvariableop_36_adam_learning_rateIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37Ё
AssignVariableOp_37AssignVariableOpassignvariableop_37_totalIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38Ё
AssignVariableOp_38AssignVariableOpassignvariableop_38_countIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39Ѓ
AssignVariableOp_39AssignVariableOpassignvariableop_39_total_1Identity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40Ѓ
AssignVariableOp_40AssignVariableOpassignvariableop_40_count_1Identity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41Ѓ
AssignVariableOp_41AssignVariableOpassignvariableop_41_total_2Identity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42Ѓ
AssignVariableOp_42AssignVariableOpassignvariableop_42_count_2Identity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43Џ
AssignVariableOp_43AssignVariableOp'assignvariableop_43_adam_input_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44­
AssignVariableOp_44AssignVariableOp%assignvariableop_44_adam_input_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45В
AssignVariableOp_45AssignVariableOp*assignvariableop_45_adam_conv_1_2_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46А
AssignVariableOp_46AssignVariableOp(assignvariableop_46_adam_conv_1_2_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47В
AssignVariableOp_47AssignVariableOp*assignvariableop_47_adam_conv_2_1_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48А
AssignVariableOp_48AssignVariableOp(assignvariableop_48_adam_conv_2_1_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49В
AssignVariableOp_49AssignVariableOp*assignvariableop_49_adam_conv_2_2_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50А
AssignVariableOp_50AssignVariableOp(assignvariableop_50_adam_conv_2_2_bias_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51­
AssignVariableOp_51AssignVariableOp%assignvariableop_51_adam_bn_1_gamma_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52Ќ
AssignVariableOp_52AssignVariableOp$assignvariableop_52_adam_bn_1_beta_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53В
AssignVariableOp_53AssignVariableOp*assignvariableop_53_adam_conv_3_1_kernel_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54А
AssignVariableOp_54AssignVariableOp(assignvariableop_54_adam_conv_3_1_bias_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55В
AssignVariableOp_55AssignVariableOp*assignvariableop_55_adam_conv_3_2_kernel_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56А
AssignVariableOp_56AssignVariableOp(assignvariableop_56_adam_conv_3_2_bias_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57В
AssignVariableOp_57AssignVariableOp*assignvariableop_57_adam_conv_4_1_kernel_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58А
AssignVariableOp_58AssignVariableOp(assignvariableop_58_adam_conv_4_1_bias_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59В
AssignVariableOp_59AssignVariableOp*assignvariableop_59_adam_conv_4_2_kernel_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60А
AssignVariableOp_60AssignVariableOp(assignvariableop_60_adam_conv_4_2_bias_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61­
AssignVariableOp_61AssignVariableOp%assignvariableop_61_adam_bn_2_gamma_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62Ќ
AssignVariableOp_62AssignVariableOp$assignvariableop_62_adam_bn_2_beta_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63Б
AssignVariableOp_63AssignVariableOp)assignvariableop_63_adam_dense_1_kernel_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64Џ
AssignVariableOp_64AssignVariableOp'assignvariableop_64_adam_dense_1_bias_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65Е
AssignVariableOp_65AssignVariableOp-assignvariableop_65_adam_dense_nxp_2_kernel_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66Г
AssignVariableOp_66AssignVariableOp+assignvariableop_66_adam_dense_nxp_2_bias_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67Е
AssignVariableOp_67AssignVariableOp-assignvariableop_67_adam_dense_nxp_3_kernel_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68Г
AssignVariableOp_68AssignVariableOp+assignvariableop_68_adam_dense_nxp_3_bias_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69Б
AssignVariableOp_69AssignVariableOp)assignvariableop_69_adam_targets_kernel_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70Џ
AssignVariableOp_70AssignVariableOp'assignvariableop_70_adam_targets_bias_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71Џ
AssignVariableOp_71AssignVariableOp'assignvariableop_71_adam_input_kernel_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72­
AssignVariableOp_72AssignVariableOp%assignvariableop_72_adam_input_bias_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73В
AssignVariableOp_73AssignVariableOp*assignvariableop_73_adam_conv_1_2_kernel_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74А
AssignVariableOp_74AssignVariableOp(assignvariableop_74_adam_conv_1_2_bias_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75В
AssignVariableOp_75AssignVariableOp*assignvariableop_75_adam_conv_2_1_kernel_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76А
AssignVariableOp_76AssignVariableOp(assignvariableop_76_adam_conv_2_1_bias_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77В
AssignVariableOp_77AssignVariableOp*assignvariableop_77_adam_conv_2_2_kernel_vIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78А
AssignVariableOp_78AssignVariableOp(assignvariableop_78_adam_conv_2_2_bias_vIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79­
AssignVariableOp_79AssignVariableOp%assignvariableop_79_adam_bn_1_gamma_vIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80Ќ
AssignVariableOp_80AssignVariableOp$assignvariableop_80_adam_bn_1_beta_vIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81В
AssignVariableOp_81AssignVariableOp*assignvariableop_81_adam_conv_3_1_kernel_vIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82А
AssignVariableOp_82AssignVariableOp(assignvariableop_82_adam_conv_3_1_bias_vIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83В
AssignVariableOp_83AssignVariableOp*assignvariableop_83_adam_conv_3_2_kernel_vIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84А
AssignVariableOp_84AssignVariableOp(assignvariableop_84_adam_conv_3_2_bias_vIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85В
AssignVariableOp_85AssignVariableOp*assignvariableop_85_adam_conv_4_1_kernel_vIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_85n
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:2
Identity_86А
AssignVariableOp_86AssignVariableOp(assignvariableop_86_adam_conv_4_1_bias_vIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_86n
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:2
Identity_87В
AssignVariableOp_87AssignVariableOp*assignvariableop_87_adam_conv_4_2_kernel_vIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_87n
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:2
Identity_88А
AssignVariableOp_88AssignVariableOp(assignvariableop_88_adam_conv_4_2_bias_vIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_88n
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:2
Identity_89­
AssignVariableOp_89AssignVariableOp%assignvariableop_89_adam_bn_2_gamma_vIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_89n
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:2
Identity_90Ќ
AssignVariableOp_90AssignVariableOp$assignvariableop_90_adam_bn_2_beta_vIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_90n
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:2
Identity_91Б
AssignVariableOp_91AssignVariableOp)assignvariableop_91_adam_dense_1_kernel_vIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_91n
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:2
Identity_92Џ
AssignVariableOp_92AssignVariableOp'assignvariableop_92_adam_dense_1_bias_vIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_92n
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:2
Identity_93Е
AssignVariableOp_93AssignVariableOp-assignvariableop_93_adam_dense_nxp_2_kernel_vIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_93n
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:2
Identity_94Г
AssignVariableOp_94AssignVariableOp+assignvariableop_94_adam_dense_nxp_2_bias_vIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_94n
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:2
Identity_95Е
AssignVariableOp_95AssignVariableOp-assignvariableop_95_adam_dense_nxp_3_kernel_vIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_95n
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:2
Identity_96Г
AssignVariableOp_96AssignVariableOp+assignvariableop_96_adam_dense_nxp_3_bias_vIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_96n
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:2
Identity_97Б
AssignVariableOp_97AssignVariableOp)assignvariableop_97_adam_targets_kernel_vIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_97n
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:2
Identity_98Џ
AssignVariableOp_98AssignVariableOp'assignvariableop_98_adam_targets_bias_vIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_989
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpр
Identity_99Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_99е
Identity_100IdentityIdentity_99:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98*
T0*
_output_shapes
: 2
Identity_100"%
identity_100Identity_100:output:0*н
_input_shapesЫ
Ш: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_98:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
­
Ў
?__inference_BN_1_layer_call_and_return_conditional_losses_32832

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<2
FusedBatchNormV3Т
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueЮ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ё
]
A__inference_Pool_1_layer_call_and_return_conditional_losses_32760

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
љ

?__inference_BN_2_layer_call_and_return_conditional_losses_32938

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3ь
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs

ќ
C__inference_Conv_2-1_layer_call_and_return_conditional_losses_33107

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЄ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџhh*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџhh2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџhh2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:џџџџџџџџџhh2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџjj: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџjj
 
_user_specified_nameinputs
Д
d
E__inference_Drop_NXP_2_layer_call_and_return_conditional_losses_33483

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeЕ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yП
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
і
c
E__inference_Drop_NXP_2_layer_call_and_return_conditional_losses_35167

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:џџџџџџџџџ2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

ќ
C__inference_Conv_3-2_layer_call_and_return_conditional_losses_34910

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOpЄ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ// *
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ// 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ// 2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:џџџџџџџџџ// 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ11 : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ11 
 
_user_specified_nameinputs
х
Ў
?__inference_BN_2_layer_call_and_return_conditional_losses_33568

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1и
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ : : : : :*
epsilon%o:*
exponential_avg_factor%
з#<2
FusedBatchNormV3Т
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueЮ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1ў
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Ф

(__inference_Conv_2-1_layer_call_fn_34715

inputs!
unknown:
	unknown_0:
identityЂStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџhh*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Conv_2-1_layer_call_and_return_conditional_losses_331072
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџhh2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџjj: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџjj
 
_user_specified_nameinputs
h
а
G__inference_sequential_1_layer_call_and_return_conditional_losses_33864

inputs%
input_33777:
input_33779:(
conv_1_2_33782:
conv_1_2_33784:(
conv_2_1_33788:
conv_2_1_33790:(
conv_2_2_33793:
conv_2_2_33795:

bn_1_33798:

bn_1_33800:

bn_1_33802:

bn_1_33804:(
conv_3_1_33808: 
conv_3_1_33810: (
conv_3_2_33813:  
conv_3_2_33815: (
conv_4_1_33819:  
conv_4_1_33821: (
conv_4_2_33824:  
conv_4_2_33826: 

bn_2_33829: 

bn_2_33831: 

bn_2_33833: 

bn_2_33835: !
dense_1_33840:
 
dense_1_33842:	%
dense_nxp_2_33846:
 
dense_nxp_2_33848:	%
dense_nxp_3_33852:
 
dense_nxp_3_33854:	 
targets_33858:	
targets_33860:
identityЂBN_1/StatefulPartitionedCallЂBN_2/StatefulPartitionedCallЂ Conv_1-2/StatefulPartitionedCallЂ Conv_2-1/StatefulPartitionedCallЂ Conv_2-2/StatefulPartitionedCallЂ Conv_3-1/StatefulPartitionedCallЂ Conv_3-2/StatefulPartitionedCallЂ Conv_4-1/StatefulPartitionedCallЂ Conv_4-2/StatefulPartitionedCallЂDense_1/StatefulPartitionedCallЂ#Dense_NXP_2/StatefulPartitionedCallЂ#Dense_NXP_3/StatefulPartitionedCallЂDrop_1/StatefulPartitionedCallЂ"Drop_NXP_2/StatefulPartitionedCallЂ"Drop_NXP_3/StatefulPartitionedCallЂinput/StatefulPartitionedCallЂtargets/StatefulPartitionedCall
input/StatefulPartitionedCallStatefulPartitionedCallinputsinput_33777input_33779*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџжж*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_input_layer_call_and_return_conditional_losses_330722
input/StatefulPartitionedCallО
 Conv_1-2/StatefulPartitionedCallStatefulPartitionedCall&input/StatefulPartitionedCall:output:0conv_1_2_33782conv_1_2_33784*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџдд*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Conv_1-2_layer_call_and_return_conditional_losses_330892"
 Conv_1-2/StatefulPartitionedCallћ
Pool_1/PartitionedCallPartitionedCall)Conv_1-2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџjj* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_Pool_1_layer_call_and_return_conditional_losses_327602
Pool_1/PartitionedCallЕ
 Conv_2-1/StatefulPartitionedCallStatefulPartitionedCallPool_1/PartitionedCall:output:0conv_2_1_33788conv_2_1_33790*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџhh*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Conv_2-1_layer_call_and_return_conditional_losses_331072"
 Conv_2-1/StatefulPartitionedCallП
 Conv_2-2/StatefulPartitionedCallStatefulPartitionedCall)Conv_2-1/StatefulPartitionedCall:output:0conv_2_2_33793conv_2_2_33795*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџff*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Conv_2-2_layer_call_and_return_conditional_losses_331242"
 Conv_2-2/StatefulPartitionedCallХ
BN_1/StatefulPartitionedCallStatefulPartitionedCall)Conv_2-2/StatefulPartitionedCall:output:0
bn_1_33798
bn_1_33800
bn_1_33802
bn_1_33804*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџff*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_BN_1_layer_call_and_return_conditional_losses_336522
BN_1/StatefulPartitionedCallї
Pool_2/PartitionedCallPartitionedCall%BN_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ33* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_Pool_2_layer_call_and_return_conditional_losses_328982
Pool_2/PartitionedCallЕ
 Conv_3-1/StatefulPartitionedCallStatefulPartitionedCallPool_2/PartitionedCall:output:0conv_3_1_33808conv_3_1_33810*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ11 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Conv_3-1_layer_call_and_return_conditional_losses_331692"
 Conv_3-1/StatefulPartitionedCallП
 Conv_3-2/StatefulPartitionedCallStatefulPartitionedCall)Conv_3-1/StatefulPartitionedCall:output:0conv_3_2_33813conv_3_2_33815*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ// *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Conv_3-2_layer_call_and_return_conditional_losses_331862"
 Conv_3-2/StatefulPartitionedCallћ
Pool_3/PartitionedCallPartitionedCall)Conv_3-2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_Pool_3_layer_call_and_return_conditional_losses_329102
Pool_3/PartitionedCallЕ
 Conv_4-1/StatefulPartitionedCallStatefulPartitionedCallPool_3/PartitionedCall:output:0conv_4_1_33819conv_4_1_33821*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Conv_4-1_layer_call_and_return_conditional_losses_332042"
 Conv_4-1/StatefulPartitionedCallП
 Conv_4-2/StatefulPartitionedCallStatefulPartitionedCall)Conv_4-1/StatefulPartitionedCall:output:0conv_4_2_33824conv_4_2_33826*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Conv_4-2_layer_call_and_return_conditional_losses_332212"
 Conv_4-2/StatefulPartitionedCallХ
BN_2/StatefulPartitionedCallStatefulPartitionedCall)Conv_4-2/StatefulPartitionedCall:output:0
bn_2_33829
bn_2_33831
bn_2_33833
bn_2_33835*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_BN_2_layer_call_and_return_conditional_losses_335682
BN_2/StatefulPartitionedCallї
Pool_4/PartitionedCallPartitionedCall%BN_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ		 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_Pool_4_layer_call_and_return_conditional_losses_330482
Pool_4/PartitionedCallѓ
Flatten_1/PartitionedCallPartitionedCallPool_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Flatten_1_layer_call_and_return_conditional_losses_332612
Flatten_1/PartitionedCallЌ
Dense_1/StatefulPartitionedCallStatefulPartitionedCall"Flatten_1/PartitionedCall:output:0dense_1_33840dense_1_33842*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_Dense_1_layer_call_and_return_conditional_losses_332742!
Dense_1/StatefulPartitionedCall
Drop_1/StatefulPartitionedCallStatefulPartitionedCall(Dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_Drop_1_layer_call_and_return_conditional_losses_335162 
Drop_1/StatefulPartitionedCallХ
#Dense_NXP_2/StatefulPartitionedCallStatefulPartitionedCall'Drop_1/StatefulPartitionedCall:output:0dense_nxp_2_33846dense_nxp_2_33848*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_Dense_NXP_2_layer_call_and_return_conditional_losses_332982%
#Dense_NXP_2/StatefulPartitionedCallМ
"Drop_NXP_2/StatefulPartitionedCallStatefulPartitionedCall,Dense_NXP_2/StatefulPartitionedCall:output:0^Drop_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_Drop_NXP_2_layer_call_and_return_conditional_losses_334832$
"Drop_NXP_2/StatefulPartitionedCallЩ
#Dense_NXP_3/StatefulPartitionedCallStatefulPartitionedCall+Drop_NXP_2/StatefulPartitionedCall:output:0dense_nxp_3_33852dense_nxp_3_33854*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_Dense_NXP_3_layer_call_and_return_conditional_losses_333222%
#Dense_NXP_3/StatefulPartitionedCallР
"Drop_NXP_3/StatefulPartitionedCallStatefulPartitionedCall,Dense_NXP_3/StatefulPartitionedCall:output:0#^Drop_NXP_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_Drop_NXP_3_layer_call_and_return_conditional_losses_334502$
"Drop_NXP_3/StatefulPartitionedCallД
targets/StatefulPartitionedCallStatefulPartitionedCall+Drop_NXP_3/StatefulPartitionedCall:output:0targets_33858targets_33860*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_targets_layer_call_and_return_conditional_losses_333462!
targets/StatefulPartitionedCallЪ
IdentityIdentity(targets/StatefulPartitionedCall:output:0^BN_1/StatefulPartitionedCall^BN_2/StatefulPartitionedCall!^Conv_1-2/StatefulPartitionedCall!^Conv_2-1/StatefulPartitionedCall!^Conv_2-2/StatefulPartitionedCall!^Conv_3-1/StatefulPartitionedCall!^Conv_3-2/StatefulPartitionedCall!^Conv_4-1/StatefulPartitionedCall!^Conv_4-2/StatefulPartitionedCall ^Dense_1/StatefulPartitionedCall$^Dense_NXP_2/StatefulPartitionedCall$^Dense_NXP_3/StatefulPartitionedCall^Drop_1/StatefulPartitionedCall#^Drop_NXP_2/StatefulPartitionedCall#^Drop_NXP_3/StatefulPartitionedCall^input/StatefulPartitionedCall ^targets/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:џџџџџџџџџии: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
BN_1/StatefulPartitionedCallBN_1/StatefulPartitionedCall2<
BN_2/StatefulPartitionedCallBN_2/StatefulPartitionedCall2D
 Conv_1-2/StatefulPartitionedCall Conv_1-2/StatefulPartitionedCall2D
 Conv_2-1/StatefulPartitionedCall Conv_2-1/StatefulPartitionedCall2D
 Conv_2-2/StatefulPartitionedCall Conv_2-2/StatefulPartitionedCall2D
 Conv_3-1/StatefulPartitionedCall Conv_3-1/StatefulPartitionedCall2D
 Conv_3-2/StatefulPartitionedCall Conv_3-2/StatefulPartitionedCall2D
 Conv_4-1/StatefulPartitionedCall Conv_4-1/StatefulPartitionedCall2D
 Conv_4-2/StatefulPartitionedCall Conv_4-2/StatefulPartitionedCall2B
Dense_1/StatefulPartitionedCallDense_1/StatefulPartitionedCall2J
#Dense_NXP_2/StatefulPartitionedCall#Dense_NXP_2/StatefulPartitionedCall2J
#Dense_NXP_3/StatefulPartitionedCall#Dense_NXP_3/StatefulPartitionedCall2@
Drop_1/StatefulPartitionedCallDrop_1/StatefulPartitionedCall2H
"Drop_NXP_2/StatefulPartitionedCall"Drop_NXP_2/StatefulPartitionedCall2H
"Drop_NXP_3/StatefulPartitionedCall"Drop_NXP_3/StatefulPartitionedCall2>
input/StatefulPartitionedCallinput/StatefulPartitionedCall2B
targets/StatefulPartitionedCalltargets/StatefulPartitionedCall:Y U
1
_output_shapes
:џџџџџџџџџии
 
_user_specified_nameinputs
 
ќ
C__inference_Conv_1-2_layer_call_and_return_conditional_losses_34706

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpІ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџдд*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџдд2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:џџџџџџџџџдд2
ReluЁ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:џџџџџџџџџдд2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџжж: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:џџџџџџџџџжж
 
_user_specified_nameinputs
ў
П
$__inference_BN_1_layer_call_fn_34798

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџff*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_BN_1_layer_call_and_return_conditional_losses_336522
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџff2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџff: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџff
 
_user_specified_nameinputs
Ф

(__inference_Conv_4-1_layer_call_fn_34919

inputs!
unknown:  
	unknown_0: 
identityЂStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Conv_4-1_layer_call_and_return_conditional_losses_332042
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Щ
_
&__inference_Drop_1_layer_call_fn_35115

inputs
identityЂStatefulPartitionedCallл
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_Drop_1_layer_call_and_return_conditional_losses_335162
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Д
d
E__inference_Drop_NXP_3_layer_call_and_return_conditional_losses_35226

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeЕ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2
dropout/GreaterEqual/yП
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

П
$__inference_BN_1_layer_call_fn_34785

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџff*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_BN_1_layer_call_and_return_conditional_losses_331472
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџff2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџff: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџff
 
_user_specified_nameinputs
Ц
П
$__inference_BN_2_layer_call_fn_34976

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identityЂStatefulPartitionedCallЄ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_BN_2_layer_call_and_return_conditional_losses_329822
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs
Н
B
&__inference_Drop_1_layer_call_fn_35110

inputs
identityУ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_Drop_1_layer_call_and_return_conditional_losses_332852
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ж

і
B__inference_Dense_1_layer_call_and_return_conditional_losses_35105

inputs2
matmul_readvariableop_resource:
 .
biasadd_readvariableop_resource:	
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
 *
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
­
Ў
?__inference_BN_2_layer_call_and_return_conditional_losses_32982

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : :*
epsilon%o:*
exponential_avg_factor%
з#<2
FusedBatchNormV3Т
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueЮ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs

љ
@__inference_input_layer_call_and_return_conditional_losses_34686

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpІ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџжж*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџжж2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:џџџџџџџџџжж2
ReluЁ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:џџџџџџџџџжж2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџии: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:џџџџџџџџџии
 
_user_specified_nameinputs
Б

?__inference_BN_2_layer_call_and_return_conditional_losses_33244

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3к
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
А

њ
F__inference_Dense_NXP_3_layer_call_and_return_conditional_losses_33322

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAddY
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Tanh
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

ќ
C__inference_Conv_3-1_layer_call_and_return_conditional_losses_34890

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpЄ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ11 *
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ11 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ11 2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:џџџџџџџџџ11 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ33: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ33
 
_user_specified_nameinputs

П
$__inference_BN_2_layer_call_fn_34989

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_BN_2_layer_call_and_return_conditional_losses_332442
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Љ

+__inference_Dense_NXP_2_layer_call_fn_35141

inputs
unknown:

	unknown_0:	
identityЂStatefulPartitionedCallњ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_Dense_NXP_2_layer_call_and_return_conditional_losses_332982
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Б

?__inference_BN_1_layer_call_and_return_conditional_losses_33147

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџff:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3к
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:џџџџџџџџџff2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџff: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:џџџџџџџџџff
 
_user_specified_nameinputs
Ц
П
$__inference_BN_1_layer_call_fn_34772

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityЂStatefulPartitionedCallЄ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_BN_1_layer_call_and_return_conditional_losses_328322
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ш
П
$__inference_BN_1_layer_call_fn_34759

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityЂStatefulPartitionedCallІ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_BN_1_layer_call_and_return_conditional_losses_327882
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
А
`
A__inference_Drop_1_layer_call_and_return_conditional_losses_35132

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeЕ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yП
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

ќ
C__inference_Conv_4-2_layer_call_and_return_conditional_losses_33221

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOpЄ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
х
Ў
?__inference_BN_1_layer_call_and_return_conditional_losses_34870

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1и
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџff:::::*
epsilon%o:*
exponential_avg_factor%
з#<2
FusedBatchNormV3Т
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueЮ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1ў
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:џџџџџџџџџff2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџff: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:џџџџџџџџџff
 
_user_specified_nameinputs
Ё

'__inference_Dense_1_layer_call_fn_35094

inputs
unknown:
 
	unknown_0:	
identityЂStatefulPartitionedCallі
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_Dense_1_layer_call_and_return_conditional_losses_332742
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
­
Ў
?__inference_BN_1_layer_call_and_return_conditional_losses_34834

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
exponential_avg_factor%
з#<2
FusedBatchNormV3Т
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueЮ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Є

є
B__inference_targets_layer_call_and_return_conditional_losses_35246

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Tanh
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Б

?__inference_BN_2_layer_call_and_return_conditional_losses_35056

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3к
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
ђ
_
A__inference_Drop_1_layer_call_and_return_conditional_losses_33285

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:џџџџџџџџџ2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
О
ф
G__inference_sequential_1_layer_call_and_return_conditional_losses_34520

inputs>
$input_conv2d_readvariableop_resource:3
%input_biasadd_readvariableop_resource:A
'conv_1_2_conv2d_readvariableop_resource:6
(conv_1_2_biasadd_readvariableop_resource:A
'conv_2_1_conv2d_readvariableop_resource:6
(conv_2_1_biasadd_readvariableop_resource:A
'conv_2_2_conv2d_readvariableop_resource:6
(conv_2_2_biasadd_readvariableop_resource:*
bn_1_readvariableop_resource:,
bn_1_readvariableop_1_resource:;
-bn_1_fusedbatchnormv3_readvariableop_resource:=
/bn_1_fusedbatchnormv3_readvariableop_1_resource:A
'conv_3_1_conv2d_readvariableop_resource: 6
(conv_3_1_biasadd_readvariableop_resource: A
'conv_3_2_conv2d_readvariableop_resource:  6
(conv_3_2_biasadd_readvariableop_resource: A
'conv_4_1_conv2d_readvariableop_resource:  6
(conv_4_1_biasadd_readvariableop_resource: A
'conv_4_2_conv2d_readvariableop_resource:  6
(conv_4_2_biasadd_readvariableop_resource: *
bn_2_readvariableop_resource: ,
bn_2_readvariableop_1_resource: ;
-bn_2_fusedbatchnormv3_readvariableop_resource: =
/bn_2_fusedbatchnormv3_readvariableop_1_resource: :
&dense_1_matmul_readvariableop_resource:
 6
'dense_1_biasadd_readvariableop_resource:	>
*dense_nxp_2_matmul_readvariableop_resource:
:
+dense_nxp_2_biasadd_readvariableop_resource:	>
*dense_nxp_3_matmul_readvariableop_resource:
:
+dense_nxp_3_biasadd_readvariableop_resource:	9
&targets_matmul_readvariableop_resource:	5
'targets_biasadd_readvariableop_resource:
identityЂ$BN_1/FusedBatchNormV3/ReadVariableOpЂ&BN_1/FusedBatchNormV3/ReadVariableOp_1ЂBN_1/ReadVariableOpЂBN_1/ReadVariableOp_1Ђ$BN_2/FusedBatchNormV3/ReadVariableOpЂ&BN_2/FusedBatchNormV3/ReadVariableOp_1ЂBN_2/ReadVariableOpЂBN_2/ReadVariableOp_1ЂConv_1-2/BiasAdd/ReadVariableOpЂConv_1-2/Conv2D/ReadVariableOpЂConv_2-1/BiasAdd/ReadVariableOpЂConv_2-1/Conv2D/ReadVariableOpЂConv_2-2/BiasAdd/ReadVariableOpЂConv_2-2/Conv2D/ReadVariableOpЂConv_3-1/BiasAdd/ReadVariableOpЂConv_3-1/Conv2D/ReadVariableOpЂConv_3-2/BiasAdd/ReadVariableOpЂConv_3-2/Conv2D/ReadVariableOpЂConv_4-1/BiasAdd/ReadVariableOpЂConv_4-1/Conv2D/ReadVariableOpЂConv_4-2/BiasAdd/ReadVariableOpЂConv_4-2/Conv2D/ReadVariableOpЂDense_1/BiasAdd/ReadVariableOpЂDense_1/MatMul/ReadVariableOpЂ"Dense_NXP_2/BiasAdd/ReadVariableOpЂ!Dense_NXP_2/MatMul/ReadVariableOpЂ"Dense_NXP_3/BiasAdd/ReadVariableOpЂ!Dense_NXP_3/MatMul/ReadVariableOpЂinput/BiasAdd/ReadVariableOpЂinput/Conv2D/ReadVariableOpЂtargets/BiasAdd/ReadVariableOpЂtargets/MatMul/ReadVariableOpЇ
input/Conv2D/ReadVariableOpReadVariableOp$input_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
input/Conv2D/ReadVariableOpИ
input/Conv2DConv2Dinputs#input/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџжж*
paddingVALID*
strides
2
input/Conv2D
input/BiasAdd/ReadVariableOpReadVariableOp%input_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
input/BiasAdd/ReadVariableOpЂ
input/BiasAddBiasAddinput/Conv2D:output:0$input/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџжж2
input/BiasAddt

input/ReluReluinput/BiasAdd:output:0*
T0*1
_output_shapes
:џџџџџџџџџжж2

input/ReluА
Conv_1-2/Conv2D/ReadVariableOpReadVariableOp'conv_1_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
Conv_1-2/Conv2D/ReadVariableOpг
Conv_1-2/Conv2DConv2Dinput/Relu:activations:0&Conv_1-2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџдд*
paddingVALID*
strides
2
Conv_1-2/Conv2DЇ
Conv_1-2/BiasAdd/ReadVariableOpReadVariableOp(conv_1_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
Conv_1-2/BiasAdd/ReadVariableOpЎ
Conv_1-2/BiasAddBiasAddConv_1-2/Conv2D:output:0'Conv_1-2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџдд2
Conv_1-2/BiasAdd}
Conv_1-2/ReluReluConv_1-2/BiasAdd:output:0*
T0*1
_output_shapes
:џџџџџџџџџдд2
Conv_1-2/ReluЕ
Pool_1/MaxPoolMaxPoolConv_1-2/Relu:activations:0*/
_output_shapes
:џџџџџџџџџjj*
ksize
*
paddingVALID*
strides
2
Pool_1/MaxPoolА
Conv_2-1/Conv2D/ReadVariableOpReadVariableOp'conv_2_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
Conv_2-1/Conv2D/ReadVariableOpа
Conv_2-1/Conv2DConv2DPool_1/MaxPool:output:0&Conv_2-1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџhh*
paddingVALID*
strides
2
Conv_2-1/Conv2DЇ
Conv_2-1/BiasAdd/ReadVariableOpReadVariableOp(conv_2_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
Conv_2-1/BiasAdd/ReadVariableOpЌ
Conv_2-1/BiasAddBiasAddConv_2-1/Conv2D:output:0'Conv_2-1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџhh2
Conv_2-1/BiasAdd{
Conv_2-1/ReluReluConv_2-1/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџhh2
Conv_2-1/ReluА
Conv_2-2/Conv2D/ReadVariableOpReadVariableOp'conv_2_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
Conv_2-2/Conv2D/ReadVariableOpд
Conv_2-2/Conv2DConv2DConv_2-1/Relu:activations:0&Conv_2-2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџff*
paddingVALID*
strides
2
Conv_2-2/Conv2DЇ
Conv_2-2/BiasAdd/ReadVariableOpReadVariableOp(conv_2_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
Conv_2-2/BiasAdd/ReadVariableOpЌ
Conv_2-2/BiasAddBiasAddConv_2-2/Conv2D:output:0'Conv_2-2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџff2
Conv_2-2/BiasAdd{
Conv_2-2/ReluReluConv_2-2/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџff2
Conv_2-2/Relu
BN_1/ReadVariableOpReadVariableOpbn_1_readvariableop_resource*
_output_shapes
:*
dtype02
BN_1/ReadVariableOp
BN_1/ReadVariableOp_1ReadVariableOpbn_1_readvariableop_1_resource*
_output_shapes
:*
dtype02
BN_1/ReadVariableOp_1Ж
$BN_1/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02&
$BN_1/FusedBatchNormV3/ReadVariableOpМ
&BN_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&BN_1/FusedBatchNormV3/ReadVariableOp_1§
BN_1/FusedBatchNormV3FusedBatchNormV3Conv_2-2/Relu:activations:0BN_1/ReadVariableOp:value:0BN_1/ReadVariableOp_1:value:0,BN_1/FusedBatchNormV3/ReadVariableOp:value:0.BN_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџff:::::*
epsilon%o:*
is_training( 2
BN_1/FusedBatchNormV3Г
Pool_2/MaxPoolMaxPoolBN_1/FusedBatchNormV3:y:0*/
_output_shapes
:џџџџџџџџџ33*
ksize
*
paddingVALID*
strides
2
Pool_2/MaxPoolА
Conv_3-1/Conv2D/ReadVariableOpReadVariableOp'conv_3_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
Conv_3-1/Conv2D/ReadVariableOpа
Conv_3-1/Conv2DConv2DPool_2/MaxPool:output:0&Conv_3-1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ11 *
paddingVALID*
strides
2
Conv_3-1/Conv2DЇ
Conv_3-1/BiasAdd/ReadVariableOpReadVariableOp(conv_3_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
Conv_3-1/BiasAdd/ReadVariableOpЌ
Conv_3-1/BiasAddBiasAddConv_3-1/Conv2D:output:0'Conv_3-1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ11 2
Conv_3-1/BiasAdd{
Conv_3-1/ReluReluConv_3-1/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ11 2
Conv_3-1/ReluА
Conv_3-2/Conv2D/ReadVariableOpReadVariableOp'conv_3_2_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02 
Conv_3-2/Conv2D/ReadVariableOpд
Conv_3-2/Conv2DConv2DConv_3-1/Relu:activations:0&Conv_3-2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ// *
paddingVALID*
strides
2
Conv_3-2/Conv2DЇ
Conv_3-2/BiasAdd/ReadVariableOpReadVariableOp(conv_3_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
Conv_3-2/BiasAdd/ReadVariableOpЌ
Conv_3-2/BiasAddBiasAddConv_3-2/Conv2D:output:0'Conv_3-2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ// 2
Conv_3-2/BiasAdd{
Conv_3-2/ReluReluConv_3-2/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ// 2
Conv_3-2/ReluЕ
Pool_3/MaxPoolMaxPoolConv_3-2/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ *
ksize
*
paddingVALID*
strides
2
Pool_3/MaxPoolА
Conv_4-1/Conv2D/ReadVariableOpReadVariableOp'conv_4_1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02 
Conv_4-1/Conv2D/ReadVariableOpа
Conv_4-1/Conv2DConv2DPool_3/MaxPool:output:0&Conv_4-1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingVALID*
strides
2
Conv_4-1/Conv2DЇ
Conv_4-1/BiasAdd/ReadVariableOpReadVariableOp(conv_4_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
Conv_4-1/BiasAdd/ReadVariableOpЌ
Conv_4-1/BiasAddBiasAddConv_4-1/Conv2D:output:0'Conv_4-1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
Conv_4-1/BiasAdd{
Conv_4-1/ReluReluConv_4-1/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
Conv_4-1/ReluА
Conv_4-2/Conv2D/ReadVariableOpReadVariableOp'conv_4_2_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02 
Conv_4-2/Conv2D/ReadVariableOpд
Conv_4-2/Conv2DConv2DConv_4-1/Relu:activations:0&Conv_4-2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingVALID*
strides
2
Conv_4-2/Conv2DЇ
Conv_4-2/BiasAdd/ReadVariableOpReadVariableOp(conv_4_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
Conv_4-2/BiasAdd/ReadVariableOpЌ
Conv_4-2/BiasAddBiasAddConv_4-2/Conv2D:output:0'Conv_4-2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
Conv_4-2/BiasAdd{
Conv_4-2/ReluReluConv_4-2/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
Conv_4-2/Relu
BN_2/ReadVariableOpReadVariableOpbn_2_readvariableop_resource*
_output_shapes
: *
dtype02
BN_2/ReadVariableOp
BN_2/ReadVariableOp_1ReadVariableOpbn_2_readvariableop_1_resource*
_output_shapes
: *
dtype02
BN_2/ReadVariableOp_1Ж
$BN_2/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02&
$BN_2/FusedBatchNormV3/ReadVariableOpМ
&BN_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&BN_2/FusedBatchNormV3/ReadVariableOp_1§
BN_2/FusedBatchNormV3FusedBatchNormV3Conv_4-2/Relu:activations:0BN_2/ReadVariableOp:value:0BN_2/ReadVariableOp_1:value:0,BN_2/FusedBatchNormV3/ReadVariableOp:value:0.BN_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ : : : : :*
epsilon%o:*
is_training( 2
BN_2/FusedBatchNormV3Г
Pool_4/MaxPoolMaxPoolBN_2/FusedBatchNormV3:y:0*/
_output_shapes
:џџџџџџџџџ		 *
ksize
*
paddingVALID*
strides
2
Pool_4/MaxPools
Flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ 
  2
Flatten_1/Const
Flatten_1/ReshapeReshapePool_4/MaxPool:output:0Flatten_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ 2
Flatten_1/ReshapeЇ
Dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
 *
dtype02
Dense_1/MatMul/ReadVariableOp 
Dense_1/MatMulMatMulFlatten_1/Reshape:output:0%Dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Dense_1/MatMulЅ
Dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
Dense_1/BiasAdd/ReadVariableOpЂ
Dense_1/BiasAddBiasAddDense_1/MatMul:product:0&Dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Dense_1/BiasAddq
Dense_1/ReluReluDense_1/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Dense_1/Relu}
Drop_1/IdentityIdentityDense_1/Relu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Drop_1/IdentityГ
!Dense_NXP_2/MatMul/ReadVariableOpReadVariableOp*dense_nxp_2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02#
!Dense_NXP_2/MatMul/ReadVariableOpЊ
Dense_NXP_2/MatMulMatMulDrop_1/Identity:output:0)Dense_NXP_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Dense_NXP_2/MatMulБ
"Dense_NXP_2/BiasAdd/ReadVariableOpReadVariableOp+dense_nxp_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02$
"Dense_NXP_2/BiasAdd/ReadVariableOpВ
Dense_NXP_2/BiasAddBiasAddDense_NXP_2/MatMul:product:0*Dense_NXP_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Dense_NXP_2/BiasAdd}
Dense_NXP_2/ReluReluDense_NXP_2/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Dense_NXP_2/Relu
Drop_NXP_2/IdentityIdentityDense_NXP_2/Relu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Drop_NXP_2/IdentityГ
!Dense_NXP_3/MatMul/ReadVariableOpReadVariableOp*dense_nxp_3_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02#
!Dense_NXP_3/MatMul/ReadVariableOpЎ
Dense_NXP_3/MatMulMatMulDrop_NXP_2/Identity:output:0)Dense_NXP_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Dense_NXP_3/MatMulБ
"Dense_NXP_3/BiasAdd/ReadVariableOpReadVariableOp+dense_nxp_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02$
"Dense_NXP_3/BiasAdd/ReadVariableOpВ
Dense_NXP_3/BiasAddBiasAddDense_NXP_3/MatMul:product:0*Dense_NXP_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Dense_NXP_3/BiasAdd}
Dense_NXP_3/TanhTanhDense_NXP_3/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Dense_NXP_3/Tanh
Drop_NXP_3/IdentityIdentityDense_NXP_3/Tanh:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Drop_NXP_3/IdentityІ
targets/MatMul/ReadVariableOpReadVariableOp&targets_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
targets/MatMul/ReadVariableOpЁ
targets/MatMulMatMulDrop_NXP_3/Identity:output:0%targets/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
targets/MatMulЄ
targets/BiasAdd/ReadVariableOpReadVariableOp'targets_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
targets/BiasAdd/ReadVariableOpЁ
targets/BiasAddBiasAddtargets/MatMul:product:0&targets/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
targets/BiasAddp
targets/TanhTanhtargets/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
targets/Tanh	
IdentityIdentitytargets/Tanh:y:0%^BN_1/FusedBatchNormV3/ReadVariableOp'^BN_1/FusedBatchNormV3/ReadVariableOp_1^BN_1/ReadVariableOp^BN_1/ReadVariableOp_1%^BN_2/FusedBatchNormV3/ReadVariableOp'^BN_2/FusedBatchNormV3/ReadVariableOp_1^BN_2/ReadVariableOp^BN_2/ReadVariableOp_1 ^Conv_1-2/BiasAdd/ReadVariableOp^Conv_1-2/Conv2D/ReadVariableOp ^Conv_2-1/BiasAdd/ReadVariableOp^Conv_2-1/Conv2D/ReadVariableOp ^Conv_2-2/BiasAdd/ReadVariableOp^Conv_2-2/Conv2D/ReadVariableOp ^Conv_3-1/BiasAdd/ReadVariableOp^Conv_3-1/Conv2D/ReadVariableOp ^Conv_3-2/BiasAdd/ReadVariableOp^Conv_3-2/Conv2D/ReadVariableOp ^Conv_4-1/BiasAdd/ReadVariableOp^Conv_4-1/Conv2D/ReadVariableOp ^Conv_4-2/BiasAdd/ReadVariableOp^Conv_4-2/Conv2D/ReadVariableOp^Dense_1/BiasAdd/ReadVariableOp^Dense_1/MatMul/ReadVariableOp#^Dense_NXP_2/BiasAdd/ReadVariableOp"^Dense_NXP_2/MatMul/ReadVariableOp#^Dense_NXP_3/BiasAdd/ReadVariableOp"^Dense_NXP_3/MatMul/ReadVariableOp^input/BiasAdd/ReadVariableOp^input/Conv2D/ReadVariableOp^targets/BiasAdd/ReadVariableOp^targets/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:џџџџџџџџџии: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$BN_1/FusedBatchNormV3/ReadVariableOp$BN_1/FusedBatchNormV3/ReadVariableOp2P
&BN_1/FusedBatchNormV3/ReadVariableOp_1&BN_1/FusedBatchNormV3/ReadVariableOp_12*
BN_1/ReadVariableOpBN_1/ReadVariableOp2.
BN_1/ReadVariableOp_1BN_1/ReadVariableOp_12L
$BN_2/FusedBatchNormV3/ReadVariableOp$BN_2/FusedBatchNormV3/ReadVariableOp2P
&BN_2/FusedBatchNormV3/ReadVariableOp_1&BN_2/FusedBatchNormV3/ReadVariableOp_12*
BN_2/ReadVariableOpBN_2/ReadVariableOp2.
BN_2/ReadVariableOp_1BN_2/ReadVariableOp_12B
Conv_1-2/BiasAdd/ReadVariableOpConv_1-2/BiasAdd/ReadVariableOp2@
Conv_1-2/Conv2D/ReadVariableOpConv_1-2/Conv2D/ReadVariableOp2B
Conv_2-1/BiasAdd/ReadVariableOpConv_2-1/BiasAdd/ReadVariableOp2@
Conv_2-1/Conv2D/ReadVariableOpConv_2-1/Conv2D/ReadVariableOp2B
Conv_2-2/BiasAdd/ReadVariableOpConv_2-2/BiasAdd/ReadVariableOp2@
Conv_2-2/Conv2D/ReadVariableOpConv_2-2/Conv2D/ReadVariableOp2B
Conv_3-1/BiasAdd/ReadVariableOpConv_3-1/BiasAdd/ReadVariableOp2@
Conv_3-1/Conv2D/ReadVariableOpConv_3-1/Conv2D/ReadVariableOp2B
Conv_3-2/BiasAdd/ReadVariableOpConv_3-2/BiasAdd/ReadVariableOp2@
Conv_3-2/Conv2D/ReadVariableOpConv_3-2/Conv2D/ReadVariableOp2B
Conv_4-1/BiasAdd/ReadVariableOpConv_4-1/BiasAdd/ReadVariableOp2@
Conv_4-1/Conv2D/ReadVariableOpConv_4-1/Conv2D/ReadVariableOp2B
Conv_4-2/BiasAdd/ReadVariableOpConv_4-2/BiasAdd/ReadVariableOp2@
Conv_4-2/Conv2D/ReadVariableOpConv_4-2/Conv2D/ReadVariableOp2@
Dense_1/BiasAdd/ReadVariableOpDense_1/BiasAdd/ReadVariableOp2>
Dense_1/MatMul/ReadVariableOpDense_1/MatMul/ReadVariableOp2H
"Dense_NXP_2/BiasAdd/ReadVariableOp"Dense_NXP_2/BiasAdd/ReadVariableOp2F
!Dense_NXP_2/MatMul/ReadVariableOp!Dense_NXP_2/MatMul/ReadVariableOp2H
"Dense_NXP_3/BiasAdd/ReadVariableOp"Dense_NXP_3/BiasAdd/ReadVariableOp2F
!Dense_NXP_3/MatMul/ReadVariableOp!Dense_NXP_3/MatMul/ReadVariableOp2<
input/BiasAdd/ReadVariableOpinput/BiasAdd/ReadVariableOp2:
input/Conv2D/ReadVariableOpinput/Conv2D/ReadVariableOp2@
targets/BiasAdd/ReadVariableOptargets/BiasAdd/ReadVariableOp2>
targets/MatMul/ReadVariableOptargets/MatMul/ReadVariableOp:Y U
1
_output_shapes
:џџџџџџџџџии
 
_user_specified_nameinputs
Є

є
B__inference_targets_layer_call_and_return_conditional_losses_33346

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Tanh
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
А
`
A__inference_Drop_1_layer_call_and_return_conditional_losses_33516

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeЕ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yП
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ф

(__inference_Conv_3-1_layer_call_fn_34879

inputs!
unknown: 
	unknown_0: 
identityЂStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ11 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Conv_3-1_layer_call_and_return_conditional_losses_331692
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ11 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ33: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ33
 
_user_specified_nameinputs
ўШ

 __inference__wrapped_model_32754
input_inputK
1sequential_1_input_conv2d_readvariableop_resource:@
2sequential_1_input_biasadd_readvariableop_resource:N
4sequential_1_conv_1_2_conv2d_readvariableop_resource:C
5sequential_1_conv_1_2_biasadd_readvariableop_resource:N
4sequential_1_conv_2_1_conv2d_readvariableop_resource:C
5sequential_1_conv_2_1_biasadd_readvariableop_resource:N
4sequential_1_conv_2_2_conv2d_readvariableop_resource:C
5sequential_1_conv_2_2_biasadd_readvariableop_resource:7
)sequential_1_bn_1_readvariableop_resource:9
+sequential_1_bn_1_readvariableop_1_resource:H
:sequential_1_bn_1_fusedbatchnormv3_readvariableop_resource:J
<sequential_1_bn_1_fusedbatchnormv3_readvariableop_1_resource:N
4sequential_1_conv_3_1_conv2d_readvariableop_resource: C
5sequential_1_conv_3_1_biasadd_readvariableop_resource: N
4sequential_1_conv_3_2_conv2d_readvariableop_resource:  C
5sequential_1_conv_3_2_biasadd_readvariableop_resource: N
4sequential_1_conv_4_1_conv2d_readvariableop_resource:  C
5sequential_1_conv_4_1_biasadd_readvariableop_resource: N
4sequential_1_conv_4_2_conv2d_readvariableop_resource:  C
5sequential_1_conv_4_2_biasadd_readvariableop_resource: 7
)sequential_1_bn_2_readvariableop_resource: 9
+sequential_1_bn_2_readvariableop_1_resource: H
:sequential_1_bn_2_fusedbatchnormv3_readvariableop_resource: J
<sequential_1_bn_2_fusedbatchnormv3_readvariableop_1_resource: G
3sequential_1_dense_1_matmul_readvariableop_resource:
 C
4sequential_1_dense_1_biasadd_readvariableop_resource:	K
7sequential_1_dense_nxp_2_matmul_readvariableop_resource:
G
8sequential_1_dense_nxp_2_biasadd_readvariableop_resource:	K
7sequential_1_dense_nxp_3_matmul_readvariableop_resource:
G
8sequential_1_dense_nxp_3_biasadd_readvariableop_resource:	F
3sequential_1_targets_matmul_readvariableop_resource:	B
4sequential_1_targets_biasadd_readvariableop_resource:
identityЂ1sequential_1/BN_1/FusedBatchNormV3/ReadVariableOpЂ3sequential_1/BN_1/FusedBatchNormV3/ReadVariableOp_1Ђ sequential_1/BN_1/ReadVariableOpЂ"sequential_1/BN_1/ReadVariableOp_1Ђ1sequential_1/BN_2/FusedBatchNormV3/ReadVariableOpЂ3sequential_1/BN_2/FusedBatchNormV3/ReadVariableOp_1Ђ sequential_1/BN_2/ReadVariableOpЂ"sequential_1/BN_2/ReadVariableOp_1Ђ,sequential_1/Conv_1-2/BiasAdd/ReadVariableOpЂ+sequential_1/Conv_1-2/Conv2D/ReadVariableOpЂ,sequential_1/Conv_2-1/BiasAdd/ReadVariableOpЂ+sequential_1/Conv_2-1/Conv2D/ReadVariableOpЂ,sequential_1/Conv_2-2/BiasAdd/ReadVariableOpЂ+sequential_1/Conv_2-2/Conv2D/ReadVariableOpЂ,sequential_1/Conv_3-1/BiasAdd/ReadVariableOpЂ+sequential_1/Conv_3-1/Conv2D/ReadVariableOpЂ,sequential_1/Conv_3-2/BiasAdd/ReadVariableOpЂ+sequential_1/Conv_3-2/Conv2D/ReadVariableOpЂ,sequential_1/Conv_4-1/BiasAdd/ReadVariableOpЂ+sequential_1/Conv_4-1/Conv2D/ReadVariableOpЂ,sequential_1/Conv_4-2/BiasAdd/ReadVariableOpЂ+sequential_1/Conv_4-2/Conv2D/ReadVariableOpЂ+sequential_1/Dense_1/BiasAdd/ReadVariableOpЂ*sequential_1/Dense_1/MatMul/ReadVariableOpЂ/sequential_1/Dense_NXP_2/BiasAdd/ReadVariableOpЂ.sequential_1/Dense_NXP_2/MatMul/ReadVariableOpЂ/sequential_1/Dense_NXP_3/BiasAdd/ReadVariableOpЂ.sequential_1/Dense_NXP_3/MatMul/ReadVariableOpЂ)sequential_1/input/BiasAdd/ReadVariableOpЂ(sequential_1/input/Conv2D/ReadVariableOpЂ+sequential_1/targets/BiasAdd/ReadVariableOpЂ*sequential_1/targets/MatMul/ReadVariableOpЮ
(sequential_1/input/Conv2D/ReadVariableOpReadVariableOp1sequential_1_input_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02*
(sequential_1/input/Conv2D/ReadVariableOpф
sequential_1/input/Conv2DConv2Dinput_input0sequential_1/input/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџжж*
paddingVALID*
strides
2
sequential_1/input/Conv2DХ
)sequential_1/input/BiasAdd/ReadVariableOpReadVariableOp2sequential_1_input_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)sequential_1/input/BiasAdd/ReadVariableOpж
sequential_1/input/BiasAddBiasAdd"sequential_1/input/Conv2D:output:01sequential_1/input/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџжж2
sequential_1/input/BiasAdd
sequential_1/input/ReluRelu#sequential_1/input/BiasAdd:output:0*
T0*1
_output_shapes
:џџџџџџџџџжж2
sequential_1/input/Reluз
+sequential_1/Conv_1-2/Conv2D/ReadVariableOpReadVariableOp4sequential_1_conv_1_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02-
+sequential_1/Conv_1-2/Conv2D/ReadVariableOp
sequential_1/Conv_1-2/Conv2DConv2D%sequential_1/input/Relu:activations:03sequential_1/Conv_1-2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџдд*
paddingVALID*
strides
2
sequential_1/Conv_1-2/Conv2DЮ
,sequential_1/Conv_1-2/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_conv_1_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_1/Conv_1-2/BiasAdd/ReadVariableOpт
sequential_1/Conv_1-2/BiasAddBiasAdd%sequential_1/Conv_1-2/Conv2D:output:04sequential_1/Conv_1-2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџдд2
sequential_1/Conv_1-2/BiasAddЄ
sequential_1/Conv_1-2/ReluRelu&sequential_1/Conv_1-2/BiasAdd:output:0*
T0*1
_output_shapes
:џџџџџџџџџдд2
sequential_1/Conv_1-2/Reluм
sequential_1/Pool_1/MaxPoolMaxPool(sequential_1/Conv_1-2/Relu:activations:0*/
_output_shapes
:џџџџџџџџџjj*
ksize
*
paddingVALID*
strides
2
sequential_1/Pool_1/MaxPoolз
+sequential_1/Conv_2-1/Conv2D/ReadVariableOpReadVariableOp4sequential_1_conv_2_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02-
+sequential_1/Conv_2-1/Conv2D/ReadVariableOp
sequential_1/Conv_2-1/Conv2DConv2D$sequential_1/Pool_1/MaxPool:output:03sequential_1/Conv_2-1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџhh*
paddingVALID*
strides
2
sequential_1/Conv_2-1/Conv2DЮ
,sequential_1/Conv_2-1/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_conv_2_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_1/Conv_2-1/BiasAdd/ReadVariableOpр
sequential_1/Conv_2-1/BiasAddBiasAdd%sequential_1/Conv_2-1/Conv2D:output:04sequential_1/Conv_2-1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџhh2
sequential_1/Conv_2-1/BiasAddЂ
sequential_1/Conv_2-1/ReluRelu&sequential_1/Conv_2-1/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџhh2
sequential_1/Conv_2-1/Reluз
+sequential_1/Conv_2-2/Conv2D/ReadVariableOpReadVariableOp4sequential_1_conv_2_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02-
+sequential_1/Conv_2-2/Conv2D/ReadVariableOp
sequential_1/Conv_2-2/Conv2DConv2D(sequential_1/Conv_2-1/Relu:activations:03sequential_1/Conv_2-2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџff*
paddingVALID*
strides
2
sequential_1/Conv_2-2/Conv2DЮ
,sequential_1/Conv_2-2/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_conv_2_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_1/Conv_2-2/BiasAdd/ReadVariableOpр
sequential_1/Conv_2-2/BiasAddBiasAdd%sequential_1/Conv_2-2/Conv2D:output:04sequential_1/Conv_2-2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџff2
sequential_1/Conv_2-2/BiasAddЂ
sequential_1/Conv_2-2/ReluRelu&sequential_1/Conv_2-2/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџff2
sequential_1/Conv_2-2/ReluЊ
 sequential_1/BN_1/ReadVariableOpReadVariableOp)sequential_1_bn_1_readvariableop_resource*
_output_shapes
:*
dtype02"
 sequential_1/BN_1/ReadVariableOpА
"sequential_1/BN_1/ReadVariableOp_1ReadVariableOp+sequential_1_bn_1_readvariableop_1_resource*
_output_shapes
:*
dtype02$
"sequential_1/BN_1/ReadVariableOp_1н
1sequential_1/BN_1/FusedBatchNormV3/ReadVariableOpReadVariableOp:sequential_1_bn_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_1/BN_1/FusedBatchNormV3/ReadVariableOpу
3sequential_1/BN_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<sequential_1_bn_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_1/BN_1/FusedBatchNormV3/ReadVariableOp_1и
"sequential_1/BN_1/FusedBatchNormV3FusedBatchNormV3(sequential_1/Conv_2-2/Relu:activations:0(sequential_1/BN_1/ReadVariableOp:value:0*sequential_1/BN_1/ReadVariableOp_1:value:09sequential_1/BN_1/FusedBatchNormV3/ReadVariableOp:value:0;sequential_1/BN_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџff:::::*
epsilon%o:*
is_training( 2$
"sequential_1/BN_1/FusedBatchNormV3к
sequential_1/Pool_2/MaxPoolMaxPool&sequential_1/BN_1/FusedBatchNormV3:y:0*/
_output_shapes
:џџџџџџџџџ33*
ksize
*
paddingVALID*
strides
2
sequential_1/Pool_2/MaxPoolз
+sequential_1/Conv_3-1/Conv2D/ReadVariableOpReadVariableOp4sequential_1_conv_3_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02-
+sequential_1/Conv_3-1/Conv2D/ReadVariableOp
sequential_1/Conv_3-1/Conv2DConv2D$sequential_1/Pool_2/MaxPool:output:03sequential_1/Conv_3-1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ11 *
paddingVALID*
strides
2
sequential_1/Conv_3-1/Conv2DЮ
,sequential_1/Conv_3-1/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_conv_3_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_1/Conv_3-1/BiasAdd/ReadVariableOpр
sequential_1/Conv_3-1/BiasAddBiasAdd%sequential_1/Conv_3-1/Conv2D:output:04sequential_1/Conv_3-1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ11 2
sequential_1/Conv_3-1/BiasAddЂ
sequential_1/Conv_3-1/ReluRelu&sequential_1/Conv_3-1/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ11 2
sequential_1/Conv_3-1/Reluз
+sequential_1/Conv_3-2/Conv2D/ReadVariableOpReadVariableOp4sequential_1_conv_3_2_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02-
+sequential_1/Conv_3-2/Conv2D/ReadVariableOp
sequential_1/Conv_3-2/Conv2DConv2D(sequential_1/Conv_3-1/Relu:activations:03sequential_1/Conv_3-2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ// *
paddingVALID*
strides
2
sequential_1/Conv_3-2/Conv2DЮ
,sequential_1/Conv_3-2/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_conv_3_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_1/Conv_3-2/BiasAdd/ReadVariableOpр
sequential_1/Conv_3-2/BiasAddBiasAdd%sequential_1/Conv_3-2/Conv2D:output:04sequential_1/Conv_3-2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ// 2
sequential_1/Conv_3-2/BiasAddЂ
sequential_1/Conv_3-2/ReluRelu&sequential_1/Conv_3-2/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ// 2
sequential_1/Conv_3-2/Reluм
sequential_1/Pool_3/MaxPoolMaxPool(sequential_1/Conv_3-2/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ *
ksize
*
paddingVALID*
strides
2
sequential_1/Pool_3/MaxPoolз
+sequential_1/Conv_4-1/Conv2D/ReadVariableOpReadVariableOp4sequential_1_conv_4_1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02-
+sequential_1/Conv_4-1/Conv2D/ReadVariableOp
sequential_1/Conv_4-1/Conv2DConv2D$sequential_1/Pool_3/MaxPool:output:03sequential_1/Conv_4-1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingVALID*
strides
2
sequential_1/Conv_4-1/Conv2DЮ
,sequential_1/Conv_4-1/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_conv_4_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_1/Conv_4-1/BiasAdd/ReadVariableOpр
sequential_1/Conv_4-1/BiasAddBiasAdd%sequential_1/Conv_4-1/Conv2D:output:04sequential_1/Conv_4-1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
sequential_1/Conv_4-1/BiasAddЂ
sequential_1/Conv_4-1/ReluRelu&sequential_1/Conv_4-1/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
sequential_1/Conv_4-1/Reluз
+sequential_1/Conv_4-2/Conv2D/ReadVariableOpReadVariableOp4sequential_1_conv_4_2_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02-
+sequential_1/Conv_4-2/Conv2D/ReadVariableOp
sequential_1/Conv_4-2/Conv2DConv2D(sequential_1/Conv_4-1/Relu:activations:03sequential_1/Conv_4-2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingVALID*
strides
2
sequential_1/Conv_4-2/Conv2DЮ
,sequential_1/Conv_4-2/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_conv_4_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_1/Conv_4-2/BiasAdd/ReadVariableOpр
sequential_1/Conv_4-2/BiasAddBiasAdd%sequential_1/Conv_4-2/Conv2D:output:04sequential_1/Conv_4-2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
sequential_1/Conv_4-2/BiasAddЂ
sequential_1/Conv_4-2/ReluRelu&sequential_1/Conv_4-2/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
sequential_1/Conv_4-2/ReluЊ
 sequential_1/BN_2/ReadVariableOpReadVariableOp)sequential_1_bn_2_readvariableop_resource*
_output_shapes
: *
dtype02"
 sequential_1/BN_2/ReadVariableOpА
"sequential_1/BN_2/ReadVariableOp_1ReadVariableOp+sequential_1_bn_2_readvariableop_1_resource*
_output_shapes
: *
dtype02$
"sequential_1/BN_2/ReadVariableOp_1н
1sequential_1/BN_2/FusedBatchNormV3/ReadVariableOpReadVariableOp:sequential_1_bn_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype023
1sequential_1/BN_2/FusedBatchNormV3/ReadVariableOpу
3sequential_1/BN_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<sequential_1_bn_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype025
3sequential_1/BN_2/FusedBatchNormV3/ReadVariableOp_1и
"sequential_1/BN_2/FusedBatchNormV3FusedBatchNormV3(sequential_1/Conv_4-2/Relu:activations:0(sequential_1/BN_2/ReadVariableOp:value:0*sequential_1/BN_2/ReadVariableOp_1:value:09sequential_1/BN_2/FusedBatchNormV3/ReadVariableOp:value:0;sequential_1/BN_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ : : : : :*
epsilon%o:*
is_training( 2$
"sequential_1/BN_2/FusedBatchNormV3к
sequential_1/Pool_4/MaxPoolMaxPool&sequential_1/BN_2/FusedBatchNormV3:y:0*/
_output_shapes
:џџџџџџџџџ		 *
ksize
*
paddingVALID*
strides
2
sequential_1/Pool_4/MaxPool
sequential_1/Flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ 
  2
sequential_1/Flatten_1/ConstЫ
sequential_1/Flatten_1/ReshapeReshape$sequential_1/Pool_4/MaxPool:output:0%sequential_1/Flatten_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ 2 
sequential_1/Flatten_1/ReshapeЮ
*sequential_1/Dense_1/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
 *
dtype02,
*sequential_1/Dense_1/MatMul/ReadVariableOpд
sequential_1/Dense_1/MatMulMatMul'sequential_1/Flatten_1/Reshape:output:02sequential_1/Dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
sequential_1/Dense_1/MatMulЬ
+sequential_1/Dense_1/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+sequential_1/Dense_1/BiasAdd/ReadVariableOpж
sequential_1/Dense_1/BiasAddBiasAdd%sequential_1/Dense_1/MatMul:product:03sequential_1/Dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
sequential_1/Dense_1/BiasAdd
sequential_1/Dense_1/ReluRelu%sequential_1/Dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
sequential_1/Dense_1/ReluЄ
sequential_1/Drop_1/IdentityIdentity'sequential_1/Dense_1/Relu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ2
sequential_1/Drop_1/Identityк
.sequential_1/Dense_NXP_2/MatMul/ReadVariableOpReadVariableOp7sequential_1_dense_nxp_2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype020
.sequential_1/Dense_NXP_2/MatMul/ReadVariableOpо
sequential_1/Dense_NXP_2/MatMulMatMul%sequential_1/Drop_1/Identity:output:06sequential_1/Dense_NXP_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2!
sequential_1/Dense_NXP_2/MatMulи
/sequential_1/Dense_NXP_2/BiasAdd/ReadVariableOpReadVariableOp8sequential_1_dense_nxp_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype021
/sequential_1/Dense_NXP_2/BiasAdd/ReadVariableOpц
 sequential_1/Dense_NXP_2/BiasAddBiasAdd)sequential_1/Dense_NXP_2/MatMul:product:07sequential_1/Dense_NXP_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2"
 sequential_1/Dense_NXP_2/BiasAddЄ
sequential_1/Dense_NXP_2/ReluRelu)sequential_1/Dense_NXP_2/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
sequential_1/Dense_NXP_2/ReluА
 sequential_1/Drop_NXP_2/IdentityIdentity+sequential_1/Dense_NXP_2/Relu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ2"
 sequential_1/Drop_NXP_2/Identityк
.sequential_1/Dense_NXP_3/MatMul/ReadVariableOpReadVariableOp7sequential_1_dense_nxp_3_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype020
.sequential_1/Dense_NXP_3/MatMul/ReadVariableOpт
sequential_1/Dense_NXP_3/MatMulMatMul)sequential_1/Drop_NXP_2/Identity:output:06sequential_1/Dense_NXP_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2!
sequential_1/Dense_NXP_3/MatMulи
/sequential_1/Dense_NXP_3/BiasAdd/ReadVariableOpReadVariableOp8sequential_1_dense_nxp_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype021
/sequential_1/Dense_NXP_3/BiasAdd/ReadVariableOpц
 sequential_1/Dense_NXP_3/BiasAddBiasAdd)sequential_1/Dense_NXP_3/MatMul:product:07sequential_1/Dense_NXP_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2"
 sequential_1/Dense_NXP_3/BiasAddЄ
sequential_1/Dense_NXP_3/TanhTanh)sequential_1/Dense_NXP_3/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
sequential_1/Dense_NXP_3/TanhІ
 sequential_1/Drop_NXP_3/IdentityIdentity!sequential_1/Dense_NXP_3/Tanh:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2"
 sequential_1/Drop_NXP_3/IdentityЭ
*sequential_1/targets/MatMul/ReadVariableOpReadVariableOp3sequential_1_targets_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02,
*sequential_1/targets/MatMul/ReadVariableOpе
sequential_1/targets/MatMulMatMul)sequential_1/Drop_NXP_3/Identity:output:02sequential_1/targets/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
sequential_1/targets/MatMulЫ
+sequential_1/targets/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_targets_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+sequential_1/targets/BiasAdd/ReadVariableOpе
sequential_1/targets/BiasAddBiasAdd%sequential_1/targets/MatMul:product:03sequential_1/targets/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
sequential_1/targets/BiasAdd
sequential_1/targets/TanhTanh%sequential_1/targets/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
sequential_1/targets/TanhГ
IdentityIdentitysequential_1/targets/Tanh:y:02^sequential_1/BN_1/FusedBatchNormV3/ReadVariableOp4^sequential_1/BN_1/FusedBatchNormV3/ReadVariableOp_1!^sequential_1/BN_1/ReadVariableOp#^sequential_1/BN_1/ReadVariableOp_12^sequential_1/BN_2/FusedBatchNormV3/ReadVariableOp4^sequential_1/BN_2/FusedBatchNormV3/ReadVariableOp_1!^sequential_1/BN_2/ReadVariableOp#^sequential_1/BN_2/ReadVariableOp_1-^sequential_1/Conv_1-2/BiasAdd/ReadVariableOp,^sequential_1/Conv_1-2/Conv2D/ReadVariableOp-^sequential_1/Conv_2-1/BiasAdd/ReadVariableOp,^sequential_1/Conv_2-1/Conv2D/ReadVariableOp-^sequential_1/Conv_2-2/BiasAdd/ReadVariableOp,^sequential_1/Conv_2-2/Conv2D/ReadVariableOp-^sequential_1/Conv_3-1/BiasAdd/ReadVariableOp,^sequential_1/Conv_3-1/Conv2D/ReadVariableOp-^sequential_1/Conv_3-2/BiasAdd/ReadVariableOp,^sequential_1/Conv_3-2/Conv2D/ReadVariableOp-^sequential_1/Conv_4-1/BiasAdd/ReadVariableOp,^sequential_1/Conv_4-1/Conv2D/ReadVariableOp-^sequential_1/Conv_4-2/BiasAdd/ReadVariableOp,^sequential_1/Conv_4-2/Conv2D/ReadVariableOp,^sequential_1/Dense_1/BiasAdd/ReadVariableOp+^sequential_1/Dense_1/MatMul/ReadVariableOp0^sequential_1/Dense_NXP_2/BiasAdd/ReadVariableOp/^sequential_1/Dense_NXP_2/MatMul/ReadVariableOp0^sequential_1/Dense_NXP_3/BiasAdd/ReadVariableOp/^sequential_1/Dense_NXP_3/MatMul/ReadVariableOp*^sequential_1/input/BiasAdd/ReadVariableOp)^sequential_1/input/Conv2D/ReadVariableOp,^sequential_1/targets/BiasAdd/ReadVariableOp+^sequential_1/targets/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:џџџџџџџџџии: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2f
1sequential_1/BN_1/FusedBatchNormV3/ReadVariableOp1sequential_1/BN_1/FusedBatchNormV3/ReadVariableOp2j
3sequential_1/BN_1/FusedBatchNormV3/ReadVariableOp_13sequential_1/BN_1/FusedBatchNormV3/ReadVariableOp_12D
 sequential_1/BN_1/ReadVariableOp sequential_1/BN_1/ReadVariableOp2H
"sequential_1/BN_1/ReadVariableOp_1"sequential_1/BN_1/ReadVariableOp_12f
1sequential_1/BN_2/FusedBatchNormV3/ReadVariableOp1sequential_1/BN_2/FusedBatchNormV3/ReadVariableOp2j
3sequential_1/BN_2/FusedBatchNormV3/ReadVariableOp_13sequential_1/BN_2/FusedBatchNormV3/ReadVariableOp_12D
 sequential_1/BN_2/ReadVariableOp sequential_1/BN_2/ReadVariableOp2H
"sequential_1/BN_2/ReadVariableOp_1"sequential_1/BN_2/ReadVariableOp_12\
,sequential_1/Conv_1-2/BiasAdd/ReadVariableOp,sequential_1/Conv_1-2/BiasAdd/ReadVariableOp2Z
+sequential_1/Conv_1-2/Conv2D/ReadVariableOp+sequential_1/Conv_1-2/Conv2D/ReadVariableOp2\
,sequential_1/Conv_2-1/BiasAdd/ReadVariableOp,sequential_1/Conv_2-1/BiasAdd/ReadVariableOp2Z
+sequential_1/Conv_2-1/Conv2D/ReadVariableOp+sequential_1/Conv_2-1/Conv2D/ReadVariableOp2\
,sequential_1/Conv_2-2/BiasAdd/ReadVariableOp,sequential_1/Conv_2-2/BiasAdd/ReadVariableOp2Z
+sequential_1/Conv_2-2/Conv2D/ReadVariableOp+sequential_1/Conv_2-2/Conv2D/ReadVariableOp2\
,sequential_1/Conv_3-1/BiasAdd/ReadVariableOp,sequential_1/Conv_3-1/BiasAdd/ReadVariableOp2Z
+sequential_1/Conv_3-1/Conv2D/ReadVariableOp+sequential_1/Conv_3-1/Conv2D/ReadVariableOp2\
,sequential_1/Conv_3-2/BiasAdd/ReadVariableOp,sequential_1/Conv_3-2/BiasAdd/ReadVariableOp2Z
+sequential_1/Conv_3-2/Conv2D/ReadVariableOp+sequential_1/Conv_3-2/Conv2D/ReadVariableOp2\
,sequential_1/Conv_4-1/BiasAdd/ReadVariableOp,sequential_1/Conv_4-1/BiasAdd/ReadVariableOp2Z
+sequential_1/Conv_4-1/Conv2D/ReadVariableOp+sequential_1/Conv_4-1/Conv2D/ReadVariableOp2\
,sequential_1/Conv_4-2/BiasAdd/ReadVariableOp,sequential_1/Conv_4-2/BiasAdd/ReadVariableOp2Z
+sequential_1/Conv_4-2/Conv2D/ReadVariableOp+sequential_1/Conv_4-2/Conv2D/ReadVariableOp2Z
+sequential_1/Dense_1/BiasAdd/ReadVariableOp+sequential_1/Dense_1/BiasAdd/ReadVariableOp2X
*sequential_1/Dense_1/MatMul/ReadVariableOp*sequential_1/Dense_1/MatMul/ReadVariableOp2b
/sequential_1/Dense_NXP_2/BiasAdd/ReadVariableOp/sequential_1/Dense_NXP_2/BiasAdd/ReadVariableOp2`
.sequential_1/Dense_NXP_2/MatMul/ReadVariableOp.sequential_1/Dense_NXP_2/MatMul/ReadVariableOp2b
/sequential_1/Dense_NXP_3/BiasAdd/ReadVariableOp/sequential_1/Dense_NXP_3/BiasAdd/ReadVariableOp2`
.sequential_1/Dense_NXP_3/MatMul/ReadVariableOp.sequential_1/Dense_NXP_3/MatMul/ReadVariableOp2V
)sequential_1/input/BiasAdd/ReadVariableOp)sequential_1/input/BiasAdd/ReadVariableOp2T
(sequential_1/input/Conv2D/ReadVariableOp(sequential_1/input/Conv2D/ReadVariableOp2Z
+sequential_1/targets/BiasAdd/ReadVariableOp+sequential_1/targets/BiasAdd/ReadVariableOp2X
*sequential_1/targets/MatMul/ReadVariableOp*sequential_1/targets/MatMul/ReadVariableOp:^ Z
1
_output_shapes
:џџџџџџџџџии
%
_user_specified_nameinput_input
б
E
)__inference_Flatten_1_layer_call_fn_35079

inputs
identityЦ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Flatten_1_layer_call_and_return_conditional_losses_332612
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ		 :W S
/
_output_shapes
:џџџџџџџџџ		 
 
_user_specified_nameinputs
ц
`
D__inference_Flatten_1_layer_call_and_return_conditional_losses_35085

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ 
  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџ 2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ		 :W S
/
_output_shapes
:џџџџџџџџџ		 
 
_user_specified_nameinputs


,__inference_sequential_1_layer_call_fn_33420
input_input!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:$

unknown_11: 

unknown_12: $

unknown_13:  

unknown_14: $

unknown_15:  

unknown_16: $

unknown_17:  

unknown_18: 

unknown_19: 

unknown_20: 

unknown_21: 

unknown_22: 

unknown_23:
 

unknown_24:	

unknown_25:


unknown_26:	

unknown_27:


unknown_28:	

unknown_29:	

unknown_30:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*B
_read_only_resource_inputs$
" 	
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_333532
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:џџџџџџџџџии: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
1
_output_shapes
:џџџџџџџџџии
%
_user_specified_nameinput_input

ќ
C__inference_Conv_3-2_layer_call_and_return_conditional_losses_33186

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOpЄ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ// *
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ// 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ// 2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:џџџџџџџџџ// 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ11 : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ11 
 
_user_specified_nameinputs
 
ќ
C__inference_Conv_1-2_layer_call_and_return_conditional_losses_33089

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpІ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџдд*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџдд2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:џџџџџџџџџдд2
ReluЁ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:џџџџџџџџџдд2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџжж: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:џџџџџџџџџжж
 
_user_specified_nameinputs
Ё
]
A__inference_Pool_2_layer_call_and_return_conditional_losses_32898

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

ќ
C__inference_Conv_2-1_layer_call_and_return_conditional_losses_34726

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЄ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџhh*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџhh2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџhh2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:џџџџџџџџџhh2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџjj: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџjj
 
_user_specified_nameinputs
б
c
*__inference_Drop_NXP_3_layer_call_fn_35209

inputs
identityЂStatefulPartitionedCallп
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_Drop_NXP_3_layer_call_and_return_conditional_losses_334502
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
К
Ч'
__inference__traced_save_35566
file_prefix+
'savev2_input_kernel_read_readvariableop)
%savev2_input_bias_read_readvariableop.
*savev2_conv_1_2_kernel_read_readvariableop,
(savev2_conv_1_2_bias_read_readvariableop.
*savev2_conv_2_1_kernel_read_readvariableop,
(savev2_conv_2_1_bias_read_readvariableop.
*savev2_conv_2_2_kernel_read_readvariableop,
(savev2_conv_2_2_bias_read_readvariableop)
%savev2_bn_1_gamma_read_readvariableop(
$savev2_bn_1_beta_read_readvariableop/
+savev2_bn_1_moving_mean_read_readvariableop3
/savev2_bn_1_moving_variance_read_readvariableop.
*savev2_conv_3_1_kernel_read_readvariableop,
(savev2_conv_3_1_bias_read_readvariableop.
*savev2_conv_3_2_kernel_read_readvariableop,
(savev2_conv_3_2_bias_read_readvariableop.
*savev2_conv_4_1_kernel_read_readvariableop,
(savev2_conv_4_1_bias_read_readvariableop.
*savev2_conv_4_2_kernel_read_readvariableop,
(savev2_conv_4_2_bias_read_readvariableop)
%savev2_bn_2_gamma_read_readvariableop(
$savev2_bn_2_beta_read_readvariableop/
+savev2_bn_2_moving_mean_read_readvariableop3
/savev2_bn_2_moving_variance_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop1
-savev2_dense_nxp_2_kernel_read_readvariableop/
+savev2_dense_nxp_2_bias_read_readvariableop1
-savev2_dense_nxp_3_kernel_read_readvariableop/
+savev2_dense_nxp_3_bias_read_readvariableop-
)savev2_targets_kernel_read_readvariableop+
'savev2_targets_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop2
.savev2_adam_input_kernel_m_read_readvariableop0
,savev2_adam_input_bias_m_read_readvariableop5
1savev2_adam_conv_1_2_kernel_m_read_readvariableop3
/savev2_adam_conv_1_2_bias_m_read_readvariableop5
1savev2_adam_conv_2_1_kernel_m_read_readvariableop3
/savev2_adam_conv_2_1_bias_m_read_readvariableop5
1savev2_adam_conv_2_2_kernel_m_read_readvariableop3
/savev2_adam_conv_2_2_bias_m_read_readvariableop0
,savev2_adam_bn_1_gamma_m_read_readvariableop/
+savev2_adam_bn_1_beta_m_read_readvariableop5
1savev2_adam_conv_3_1_kernel_m_read_readvariableop3
/savev2_adam_conv_3_1_bias_m_read_readvariableop5
1savev2_adam_conv_3_2_kernel_m_read_readvariableop3
/savev2_adam_conv_3_2_bias_m_read_readvariableop5
1savev2_adam_conv_4_1_kernel_m_read_readvariableop3
/savev2_adam_conv_4_1_bias_m_read_readvariableop5
1savev2_adam_conv_4_2_kernel_m_read_readvariableop3
/savev2_adam_conv_4_2_bias_m_read_readvariableop0
,savev2_adam_bn_2_gamma_m_read_readvariableop/
+savev2_adam_bn_2_beta_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop8
4savev2_adam_dense_nxp_2_kernel_m_read_readvariableop6
2savev2_adam_dense_nxp_2_bias_m_read_readvariableop8
4savev2_adam_dense_nxp_3_kernel_m_read_readvariableop6
2savev2_adam_dense_nxp_3_bias_m_read_readvariableop4
0savev2_adam_targets_kernel_m_read_readvariableop2
.savev2_adam_targets_bias_m_read_readvariableop2
.savev2_adam_input_kernel_v_read_readvariableop0
,savev2_adam_input_bias_v_read_readvariableop5
1savev2_adam_conv_1_2_kernel_v_read_readvariableop3
/savev2_adam_conv_1_2_bias_v_read_readvariableop5
1savev2_adam_conv_2_1_kernel_v_read_readvariableop3
/savev2_adam_conv_2_1_bias_v_read_readvariableop5
1savev2_adam_conv_2_2_kernel_v_read_readvariableop3
/savev2_adam_conv_2_2_bias_v_read_readvariableop0
,savev2_adam_bn_1_gamma_v_read_readvariableop/
+savev2_adam_bn_1_beta_v_read_readvariableop5
1savev2_adam_conv_3_1_kernel_v_read_readvariableop3
/savev2_adam_conv_3_1_bias_v_read_readvariableop5
1savev2_adam_conv_3_2_kernel_v_read_readvariableop3
/savev2_adam_conv_3_2_bias_v_read_readvariableop5
1savev2_adam_conv_4_1_kernel_v_read_readvariableop3
/savev2_adam_conv_4_1_bias_v_read_readvariableop5
1savev2_adam_conv_4_2_kernel_v_read_readvariableop3
/savev2_adam_conv_4_2_bias_v_read_readvariableop0
,savev2_adam_bn_2_gamma_v_read_readvariableop/
+savev2_adam_bn_2_beta_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop8
4savev2_adam_dense_nxp_2_kernel_v_read_readvariableop6
2savev2_adam_dense_nxp_2_bias_v_read_readvariableop8
4savev2_adam_dense_nxp_3_kernel_v_read_readvariableop6
2savev2_adam_dense_nxp_3_bias_v_read_readvariableop4
0savev2_adam_targets_kernel_v_read_readvariableop2
.savev2_adam_targets_bias_v_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardІ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename8
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:d*
dtype0*7
value7B7dB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesг
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:d*
dtype0*н
valueгBаdB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesэ%
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_input_kernel_read_readvariableop%savev2_input_bias_read_readvariableop*savev2_conv_1_2_kernel_read_readvariableop(savev2_conv_1_2_bias_read_readvariableop*savev2_conv_2_1_kernel_read_readvariableop(savev2_conv_2_1_bias_read_readvariableop*savev2_conv_2_2_kernel_read_readvariableop(savev2_conv_2_2_bias_read_readvariableop%savev2_bn_1_gamma_read_readvariableop$savev2_bn_1_beta_read_readvariableop+savev2_bn_1_moving_mean_read_readvariableop/savev2_bn_1_moving_variance_read_readvariableop*savev2_conv_3_1_kernel_read_readvariableop(savev2_conv_3_1_bias_read_readvariableop*savev2_conv_3_2_kernel_read_readvariableop(savev2_conv_3_2_bias_read_readvariableop*savev2_conv_4_1_kernel_read_readvariableop(savev2_conv_4_1_bias_read_readvariableop*savev2_conv_4_2_kernel_read_readvariableop(savev2_conv_4_2_bias_read_readvariableop%savev2_bn_2_gamma_read_readvariableop$savev2_bn_2_beta_read_readvariableop+savev2_bn_2_moving_mean_read_readvariableop/savev2_bn_2_moving_variance_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop-savev2_dense_nxp_2_kernel_read_readvariableop+savev2_dense_nxp_2_bias_read_readvariableop-savev2_dense_nxp_3_kernel_read_readvariableop+savev2_dense_nxp_3_bias_read_readvariableop)savev2_targets_kernel_read_readvariableop'savev2_targets_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop.savev2_adam_input_kernel_m_read_readvariableop,savev2_adam_input_bias_m_read_readvariableop1savev2_adam_conv_1_2_kernel_m_read_readvariableop/savev2_adam_conv_1_2_bias_m_read_readvariableop1savev2_adam_conv_2_1_kernel_m_read_readvariableop/savev2_adam_conv_2_1_bias_m_read_readvariableop1savev2_adam_conv_2_2_kernel_m_read_readvariableop/savev2_adam_conv_2_2_bias_m_read_readvariableop,savev2_adam_bn_1_gamma_m_read_readvariableop+savev2_adam_bn_1_beta_m_read_readvariableop1savev2_adam_conv_3_1_kernel_m_read_readvariableop/savev2_adam_conv_3_1_bias_m_read_readvariableop1savev2_adam_conv_3_2_kernel_m_read_readvariableop/savev2_adam_conv_3_2_bias_m_read_readvariableop1savev2_adam_conv_4_1_kernel_m_read_readvariableop/savev2_adam_conv_4_1_bias_m_read_readvariableop1savev2_adam_conv_4_2_kernel_m_read_readvariableop/savev2_adam_conv_4_2_bias_m_read_readvariableop,savev2_adam_bn_2_gamma_m_read_readvariableop+savev2_adam_bn_2_beta_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop4savev2_adam_dense_nxp_2_kernel_m_read_readvariableop2savev2_adam_dense_nxp_2_bias_m_read_readvariableop4savev2_adam_dense_nxp_3_kernel_m_read_readvariableop2savev2_adam_dense_nxp_3_bias_m_read_readvariableop0savev2_adam_targets_kernel_m_read_readvariableop.savev2_adam_targets_bias_m_read_readvariableop.savev2_adam_input_kernel_v_read_readvariableop,savev2_adam_input_bias_v_read_readvariableop1savev2_adam_conv_1_2_kernel_v_read_readvariableop/savev2_adam_conv_1_2_bias_v_read_readvariableop1savev2_adam_conv_2_1_kernel_v_read_readvariableop/savev2_adam_conv_2_1_bias_v_read_readvariableop1savev2_adam_conv_2_2_kernel_v_read_readvariableop/savev2_adam_conv_2_2_bias_v_read_readvariableop,savev2_adam_bn_1_gamma_v_read_readvariableop+savev2_adam_bn_1_beta_v_read_readvariableop1savev2_adam_conv_3_1_kernel_v_read_readvariableop/savev2_adam_conv_3_1_bias_v_read_readvariableop1savev2_adam_conv_3_2_kernel_v_read_readvariableop/savev2_adam_conv_3_2_bias_v_read_readvariableop1savev2_adam_conv_4_1_kernel_v_read_readvariableop/savev2_adam_conv_4_1_bias_v_read_readvariableop1savev2_adam_conv_4_2_kernel_v_read_readvariableop/savev2_adam_conv_4_2_bias_v_read_readvariableop,savev2_adam_bn_2_gamma_v_read_readvariableop+savev2_adam_bn_2_beta_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop4savev2_adam_dense_nxp_2_kernel_v_read_readvariableop2savev2_adam_dense_nxp_2_bias_v_read_readvariableop4savev2_adam_dense_nxp_3_kernel_v_read_readvariableop2savev2_adam_dense_nxp_3_bias_v_read_readvariableop0savev2_adam_targets_kernel_v_read_readvariableop.savev2_adam_targets_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *r
dtypesh
f2d	2
SaveV2К
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesЁ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*­
_input_shapes
: ::::::::::::: : :  : :  : :  : : : : : :
 ::
::
::	:: : : : : : : : : : : ::::::::::: : :  : :  : :  : : : :
 ::
::
::	:::::::::::: : :  : :  : :  : : : :
 ::
::
::	:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
:: 	

_output_shapes
:: 


_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :&"
 
_output_shapes
:
 :!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	:  

_output_shapes
::!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,,(
&
_output_shapes
:: -

_output_shapes
::,.(
&
_output_shapes
:: /

_output_shapes
::,0(
&
_output_shapes
:: 1

_output_shapes
::,2(
&
_output_shapes
:: 3

_output_shapes
:: 4

_output_shapes
:: 5

_output_shapes
::,6(
&
_output_shapes
: : 7

_output_shapes
: :,8(
&
_output_shapes
:  : 9

_output_shapes
: :,:(
&
_output_shapes
:  : ;

_output_shapes
: :,<(
&
_output_shapes
:  : =

_output_shapes
: : >

_output_shapes
: : ?

_output_shapes
: :&@"
 
_output_shapes
:
 :!A

_output_shapes	
::&B"
 
_output_shapes
:
:!C

_output_shapes	
::&D"
 
_output_shapes
:
:!E

_output_shapes	
::%F!

_output_shapes
:	: G

_output_shapes
::,H(
&
_output_shapes
:: I

_output_shapes
::,J(
&
_output_shapes
:: K

_output_shapes
::,L(
&
_output_shapes
:: M

_output_shapes
::,N(
&
_output_shapes
:: O

_output_shapes
:: P

_output_shapes
:: Q

_output_shapes
::,R(
&
_output_shapes
: : S

_output_shapes
: :,T(
&
_output_shapes
:  : U

_output_shapes
: :,V(
&
_output_shapes
:  : W

_output_shapes
: :,X(
&
_output_shapes
:  : Y

_output_shapes
: : Z

_output_shapes
: : [

_output_shapes
: :&\"
 
_output_shapes
:
 :!]

_output_shapes	
::&^"
 
_output_shapes
:
:!_

_output_shapes	
::&`"
 
_output_shapes
:
:!a

_output_shapes	
::%b!

_output_shapes
:	: c

_output_shapes
::d

_output_shapes
: 

ќ
C__inference_Conv_4-2_layer_call_and_return_conditional_losses_34950

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOpЄ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
х
Ў
?__inference_BN_1_layer_call_and_return_conditional_losses_33652

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1и
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџff:::::*
epsilon%o:*
exponential_avg_factor%
з#<2
FusedBatchNormV3Т
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueЮ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1ў
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:џџџџџџџџџff2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџff: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:џџџџџџџџџff
 
_user_specified_nameinputs
ч

#__inference_signature_wrapper_34257
input_input!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:$

unknown_11: 

unknown_12: $

unknown_13:  

unknown_14: $

unknown_15:  

unknown_16: $

unknown_17:  

unknown_18: 

unknown_19: 

unknown_20: 

unknown_21: 

unknown_22: 

unknown_23:
 

unknown_24:	

unknown_25:


unknown_26:	

unknown_27:


unknown_28:	

unknown_29:	

unknown_30:
identityЂStatefulPartitionedCallѓ
StatefulPartitionedCallStatefulPartitionedCallinput_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*B
_read_only_resource_inputs$
" 	
 *0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__wrapped_model_327542
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:џџџџџџџџџии: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
1
_output_shapes
:џџџџџџџџџии
%
_user_specified_nameinput_input
А

њ
F__inference_Dense_NXP_3_layer_call_and_return_conditional_losses_35199

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAddY
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Tanh
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Љ

+__inference_Dense_NXP_3_layer_call_fn_35188

inputs
unknown:

	unknown_0:	
identityЂStatefulPartitionedCallњ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_Dense_NXP_3_layer_call_and_return_conditional_losses_333222
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

ќ
C__inference_Conv_4-1_layer_call_and_return_conditional_losses_34930

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOpЄ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
ЭУ
Р
G__inference_sequential_1_layer_call_and_return_conditional_losses_34666

inputs>
$input_conv2d_readvariableop_resource:3
%input_biasadd_readvariableop_resource:A
'conv_1_2_conv2d_readvariableop_resource:6
(conv_1_2_biasadd_readvariableop_resource:A
'conv_2_1_conv2d_readvariableop_resource:6
(conv_2_1_biasadd_readvariableop_resource:A
'conv_2_2_conv2d_readvariableop_resource:6
(conv_2_2_biasadd_readvariableop_resource:*
bn_1_readvariableop_resource:,
bn_1_readvariableop_1_resource:;
-bn_1_fusedbatchnormv3_readvariableop_resource:=
/bn_1_fusedbatchnormv3_readvariableop_1_resource:A
'conv_3_1_conv2d_readvariableop_resource: 6
(conv_3_1_biasadd_readvariableop_resource: A
'conv_3_2_conv2d_readvariableop_resource:  6
(conv_3_2_biasadd_readvariableop_resource: A
'conv_4_1_conv2d_readvariableop_resource:  6
(conv_4_1_biasadd_readvariableop_resource: A
'conv_4_2_conv2d_readvariableop_resource:  6
(conv_4_2_biasadd_readvariableop_resource: *
bn_2_readvariableop_resource: ,
bn_2_readvariableop_1_resource: ;
-bn_2_fusedbatchnormv3_readvariableop_resource: =
/bn_2_fusedbatchnormv3_readvariableop_1_resource: :
&dense_1_matmul_readvariableop_resource:
 6
'dense_1_biasadd_readvariableop_resource:	>
*dense_nxp_2_matmul_readvariableop_resource:
:
+dense_nxp_2_biasadd_readvariableop_resource:	>
*dense_nxp_3_matmul_readvariableop_resource:
:
+dense_nxp_3_biasadd_readvariableop_resource:	9
&targets_matmul_readvariableop_resource:	5
'targets_biasadd_readvariableop_resource:
identityЂBN_1/AssignNewValueЂBN_1/AssignNewValue_1Ђ$BN_1/FusedBatchNormV3/ReadVariableOpЂ&BN_1/FusedBatchNormV3/ReadVariableOp_1ЂBN_1/ReadVariableOpЂBN_1/ReadVariableOp_1ЂBN_2/AssignNewValueЂBN_2/AssignNewValue_1Ђ$BN_2/FusedBatchNormV3/ReadVariableOpЂ&BN_2/FusedBatchNormV3/ReadVariableOp_1ЂBN_2/ReadVariableOpЂBN_2/ReadVariableOp_1ЂConv_1-2/BiasAdd/ReadVariableOpЂConv_1-2/Conv2D/ReadVariableOpЂConv_2-1/BiasAdd/ReadVariableOpЂConv_2-1/Conv2D/ReadVariableOpЂConv_2-2/BiasAdd/ReadVariableOpЂConv_2-2/Conv2D/ReadVariableOpЂConv_3-1/BiasAdd/ReadVariableOpЂConv_3-1/Conv2D/ReadVariableOpЂConv_3-2/BiasAdd/ReadVariableOpЂConv_3-2/Conv2D/ReadVariableOpЂConv_4-1/BiasAdd/ReadVariableOpЂConv_4-1/Conv2D/ReadVariableOpЂConv_4-2/BiasAdd/ReadVariableOpЂConv_4-2/Conv2D/ReadVariableOpЂDense_1/BiasAdd/ReadVariableOpЂDense_1/MatMul/ReadVariableOpЂ"Dense_NXP_2/BiasAdd/ReadVariableOpЂ!Dense_NXP_2/MatMul/ReadVariableOpЂ"Dense_NXP_3/BiasAdd/ReadVariableOpЂ!Dense_NXP_3/MatMul/ReadVariableOpЂinput/BiasAdd/ReadVariableOpЂinput/Conv2D/ReadVariableOpЂtargets/BiasAdd/ReadVariableOpЂtargets/MatMul/ReadVariableOpЇ
input/Conv2D/ReadVariableOpReadVariableOp$input_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
input/Conv2D/ReadVariableOpИ
input/Conv2DConv2Dinputs#input/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџжж*
paddingVALID*
strides
2
input/Conv2D
input/BiasAdd/ReadVariableOpReadVariableOp%input_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
input/BiasAdd/ReadVariableOpЂ
input/BiasAddBiasAddinput/Conv2D:output:0$input/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџжж2
input/BiasAddt

input/ReluReluinput/BiasAdd:output:0*
T0*1
_output_shapes
:џџџџџџџџџжж2

input/ReluА
Conv_1-2/Conv2D/ReadVariableOpReadVariableOp'conv_1_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
Conv_1-2/Conv2D/ReadVariableOpг
Conv_1-2/Conv2DConv2Dinput/Relu:activations:0&Conv_1-2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџдд*
paddingVALID*
strides
2
Conv_1-2/Conv2DЇ
Conv_1-2/BiasAdd/ReadVariableOpReadVariableOp(conv_1_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
Conv_1-2/BiasAdd/ReadVariableOpЎ
Conv_1-2/BiasAddBiasAddConv_1-2/Conv2D:output:0'Conv_1-2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџдд2
Conv_1-2/BiasAdd}
Conv_1-2/ReluReluConv_1-2/BiasAdd:output:0*
T0*1
_output_shapes
:џџџџџџџџџдд2
Conv_1-2/ReluЕ
Pool_1/MaxPoolMaxPoolConv_1-2/Relu:activations:0*/
_output_shapes
:џџџџџџџџџjj*
ksize
*
paddingVALID*
strides
2
Pool_1/MaxPoolА
Conv_2-1/Conv2D/ReadVariableOpReadVariableOp'conv_2_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
Conv_2-1/Conv2D/ReadVariableOpа
Conv_2-1/Conv2DConv2DPool_1/MaxPool:output:0&Conv_2-1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџhh*
paddingVALID*
strides
2
Conv_2-1/Conv2DЇ
Conv_2-1/BiasAdd/ReadVariableOpReadVariableOp(conv_2_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
Conv_2-1/BiasAdd/ReadVariableOpЌ
Conv_2-1/BiasAddBiasAddConv_2-1/Conv2D:output:0'Conv_2-1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџhh2
Conv_2-1/BiasAdd{
Conv_2-1/ReluReluConv_2-1/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџhh2
Conv_2-1/ReluА
Conv_2-2/Conv2D/ReadVariableOpReadVariableOp'conv_2_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
Conv_2-2/Conv2D/ReadVariableOpд
Conv_2-2/Conv2DConv2DConv_2-1/Relu:activations:0&Conv_2-2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџff*
paddingVALID*
strides
2
Conv_2-2/Conv2DЇ
Conv_2-2/BiasAdd/ReadVariableOpReadVariableOp(conv_2_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
Conv_2-2/BiasAdd/ReadVariableOpЌ
Conv_2-2/BiasAddBiasAddConv_2-2/Conv2D:output:0'Conv_2-2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџff2
Conv_2-2/BiasAdd{
Conv_2-2/ReluReluConv_2-2/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџff2
Conv_2-2/Relu
BN_1/ReadVariableOpReadVariableOpbn_1_readvariableop_resource*
_output_shapes
:*
dtype02
BN_1/ReadVariableOp
BN_1/ReadVariableOp_1ReadVariableOpbn_1_readvariableop_1_resource*
_output_shapes
:*
dtype02
BN_1/ReadVariableOp_1Ж
$BN_1/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02&
$BN_1/FusedBatchNormV3/ReadVariableOpМ
&BN_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&BN_1/FusedBatchNormV3/ReadVariableOp_1
BN_1/FusedBatchNormV3FusedBatchNormV3Conv_2-2/Relu:activations:0BN_1/ReadVariableOp:value:0BN_1/ReadVariableOp_1:value:0,BN_1/FusedBatchNormV3/ReadVariableOp:value:0.BN_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџff:::::*
epsilon%o:*
exponential_avg_factor%
з#<2
BN_1/FusedBatchNormV3л
BN_1/AssignNewValueAssignVariableOp-bn_1_fusedbatchnormv3_readvariableop_resource"BN_1/FusedBatchNormV3:batch_mean:0%^BN_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
BN_1/AssignNewValueч
BN_1/AssignNewValue_1AssignVariableOp/bn_1_fusedbatchnormv3_readvariableop_1_resource&BN_1/FusedBatchNormV3:batch_variance:0'^BN_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
BN_1/AssignNewValue_1Г
Pool_2/MaxPoolMaxPoolBN_1/FusedBatchNormV3:y:0*/
_output_shapes
:џџџџџџџџџ33*
ksize
*
paddingVALID*
strides
2
Pool_2/MaxPoolА
Conv_3-1/Conv2D/ReadVariableOpReadVariableOp'conv_3_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
Conv_3-1/Conv2D/ReadVariableOpа
Conv_3-1/Conv2DConv2DPool_2/MaxPool:output:0&Conv_3-1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ11 *
paddingVALID*
strides
2
Conv_3-1/Conv2DЇ
Conv_3-1/BiasAdd/ReadVariableOpReadVariableOp(conv_3_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
Conv_3-1/BiasAdd/ReadVariableOpЌ
Conv_3-1/BiasAddBiasAddConv_3-1/Conv2D:output:0'Conv_3-1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ11 2
Conv_3-1/BiasAdd{
Conv_3-1/ReluReluConv_3-1/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ11 2
Conv_3-1/ReluА
Conv_3-2/Conv2D/ReadVariableOpReadVariableOp'conv_3_2_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02 
Conv_3-2/Conv2D/ReadVariableOpд
Conv_3-2/Conv2DConv2DConv_3-1/Relu:activations:0&Conv_3-2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ// *
paddingVALID*
strides
2
Conv_3-2/Conv2DЇ
Conv_3-2/BiasAdd/ReadVariableOpReadVariableOp(conv_3_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
Conv_3-2/BiasAdd/ReadVariableOpЌ
Conv_3-2/BiasAddBiasAddConv_3-2/Conv2D:output:0'Conv_3-2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ// 2
Conv_3-2/BiasAdd{
Conv_3-2/ReluReluConv_3-2/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ// 2
Conv_3-2/ReluЕ
Pool_3/MaxPoolMaxPoolConv_3-2/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ *
ksize
*
paddingVALID*
strides
2
Pool_3/MaxPoolА
Conv_4-1/Conv2D/ReadVariableOpReadVariableOp'conv_4_1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02 
Conv_4-1/Conv2D/ReadVariableOpа
Conv_4-1/Conv2DConv2DPool_3/MaxPool:output:0&Conv_4-1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingVALID*
strides
2
Conv_4-1/Conv2DЇ
Conv_4-1/BiasAdd/ReadVariableOpReadVariableOp(conv_4_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
Conv_4-1/BiasAdd/ReadVariableOpЌ
Conv_4-1/BiasAddBiasAddConv_4-1/Conv2D:output:0'Conv_4-1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
Conv_4-1/BiasAdd{
Conv_4-1/ReluReluConv_4-1/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
Conv_4-1/ReluА
Conv_4-2/Conv2D/ReadVariableOpReadVariableOp'conv_4_2_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02 
Conv_4-2/Conv2D/ReadVariableOpд
Conv_4-2/Conv2DConv2DConv_4-1/Relu:activations:0&Conv_4-2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingVALID*
strides
2
Conv_4-2/Conv2DЇ
Conv_4-2/BiasAdd/ReadVariableOpReadVariableOp(conv_4_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
Conv_4-2/BiasAdd/ReadVariableOpЌ
Conv_4-2/BiasAddBiasAddConv_4-2/Conv2D:output:0'Conv_4-2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
Conv_4-2/BiasAdd{
Conv_4-2/ReluReluConv_4-2/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
Conv_4-2/Relu
BN_2/ReadVariableOpReadVariableOpbn_2_readvariableop_resource*
_output_shapes
: *
dtype02
BN_2/ReadVariableOp
BN_2/ReadVariableOp_1ReadVariableOpbn_2_readvariableop_1_resource*
_output_shapes
: *
dtype02
BN_2/ReadVariableOp_1Ж
$BN_2/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02&
$BN_2/FusedBatchNormV3/ReadVariableOpМ
&BN_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&BN_2/FusedBatchNormV3/ReadVariableOp_1
BN_2/FusedBatchNormV3FusedBatchNormV3Conv_4-2/Relu:activations:0BN_2/ReadVariableOp:value:0BN_2/ReadVariableOp_1:value:0,BN_2/FusedBatchNormV3/ReadVariableOp:value:0.BN_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ : : : : :*
epsilon%o:*
exponential_avg_factor%
з#<2
BN_2/FusedBatchNormV3л
BN_2/AssignNewValueAssignVariableOp-bn_2_fusedbatchnormv3_readvariableop_resource"BN_2/FusedBatchNormV3:batch_mean:0%^BN_2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
BN_2/AssignNewValueч
BN_2/AssignNewValue_1AssignVariableOp/bn_2_fusedbatchnormv3_readvariableop_1_resource&BN_2/FusedBatchNormV3:batch_variance:0'^BN_2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
BN_2/AssignNewValue_1Г
Pool_4/MaxPoolMaxPoolBN_2/FusedBatchNormV3:y:0*/
_output_shapes
:џџџџџџџџџ		 *
ksize
*
paddingVALID*
strides
2
Pool_4/MaxPools
Flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ 
  2
Flatten_1/Const
Flatten_1/ReshapeReshapePool_4/MaxPool:output:0Flatten_1/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ 2
Flatten_1/ReshapeЇ
Dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
 *
dtype02
Dense_1/MatMul/ReadVariableOp 
Dense_1/MatMulMatMulFlatten_1/Reshape:output:0%Dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Dense_1/MatMulЅ
Dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
Dense_1/BiasAdd/ReadVariableOpЂ
Dense_1/BiasAddBiasAddDense_1/MatMul:product:0&Dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Dense_1/BiasAddq
Dense_1/ReluReluDense_1/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Dense_1/Reluq
Drop_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
Drop_1/dropout/Const
Drop_1/dropout/MulMulDense_1/Relu:activations:0Drop_1/dropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Drop_1/dropout/Mulv
Drop_1/dropout/ShapeShapeDense_1/Relu:activations:0*
T0*
_output_shapes
:2
Drop_1/dropout/ShapeЪ
+Drop_1/dropout/random_uniform/RandomUniformRandomUniformDrop_1/dropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype02-
+Drop_1/dropout/random_uniform/RandomUniform
Drop_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
Drop_1/dropout/GreaterEqual/yл
Drop_1/dropout/GreaterEqualGreaterEqual4Drop_1/dropout/random_uniform/RandomUniform:output:0&Drop_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Drop_1/dropout/GreaterEqual
Drop_1/dropout/CastCastDrop_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџ2
Drop_1/dropout/Cast
Drop_1/dropout/Mul_1MulDrop_1/dropout/Mul:z:0Drop_1/dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Drop_1/dropout/Mul_1Г
!Dense_NXP_2/MatMul/ReadVariableOpReadVariableOp*dense_nxp_2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02#
!Dense_NXP_2/MatMul/ReadVariableOpЊ
Dense_NXP_2/MatMulMatMulDrop_1/dropout/Mul_1:z:0)Dense_NXP_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Dense_NXP_2/MatMulБ
"Dense_NXP_2/BiasAdd/ReadVariableOpReadVariableOp+dense_nxp_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02$
"Dense_NXP_2/BiasAdd/ReadVariableOpВ
Dense_NXP_2/BiasAddBiasAddDense_NXP_2/MatMul:product:0*Dense_NXP_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Dense_NXP_2/BiasAdd}
Dense_NXP_2/ReluReluDense_NXP_2/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Dense_NXP_2/Reluy
Drop_NXP_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
Drop_NXP_2/dropout/Const­
Drop_NXP_2/dropout/MulMulDense_NXP_2/Relu:activations:0!Drop_NXP_2/dropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Drop_NXP_2/dropout/Mul
Drop_NXP_2/dropout/ShapeShapeDense_NXP_2/Relu:activations:0*
T0*
_output_shapes
:2
Drop_NXP_2/dropout/Shapeж
/Drop_NXP_2/dropout/random_uniform/RandomUniformRandomUniform!Drop_NXP_2/dropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype021
/Drop_NXP_2/dropout/random_uniform/RandomUniform
!Drop_NXP_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!Drop_NXP_2/dropout/GreaterEqual/yы
Drop_NXP_2/dropout/GreaterEqualGreaterEqual8Drop_NXP_2/dropout/random_uniform/RandomUniform:output:0*Drop_NXP_2/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2!
Drop_NXP_2/dropout/GreaterEqualЁ
Drop_NXP_2/dropout/CastCast#Drop_NXP_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџ2
Drop_NXP_2/dropout/CastЇ
Drop_NXP_2/dropout/Mul_1MulDrop_NXP_2/dropout/Mul:z:0Drop_NXP_2/dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Drop_NXP_2/dropout/Mul_1Г
!Dense_NXP_3/MatMul/ReadVariableOpReadVariableOp*dense_nxp_3_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02#
!Dense_NXP_3/MatMul/ReadVariableOpЎ
Dense_NXP_3/MatMulMatMulDrop_NXP_2/dropout/Mul_1:z:0)Dense_NXP_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Dense_NXP_3/MatMulБ
"Dense_NXP_3/BiasAdd/ReadVariableOpReadVariableOp+dense_nxp_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02$
"Dense_NXP_3/BiasAdd/ReadVariableOpВ
Dense_NXP_3/BiasAddBiasAddDense_NXP_3/MatMul:product:0*Dense_NXP_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Dense_NXP_3/BiasAdd}
Dense_NXP_3/TanhTanhDense_NXP_3/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Dense_NXP_3/Tanhy
Drop_NXP_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
Drop_NXP_3/dropout/ConstЃ
Drop_NXP_3/dropout/MulMulDense_NXP_3/Tanh:y:0!Drop_NXP_3/dropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Drop_NXP_3/dropout/Mulx
Drop_NXP_3/dropout/ShapeShapeDense_NXP_3/Tanh:y:0*
T0*
_output_shapes
:2
Drop_NXP_3/dropout/Shapeж
/Drop_NXP_3/dropout/random_uniform/RandomUniformRandomUniform!Drop_NXP_3/dropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype021
/Drop_NXP_3/dropout/random_uniform/RandomUniform
!Drop_NXP_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2#
!Drop_NXP_3/dropout/GreaterEqual/yы
Drop_NXP_3/dropout/GreaterEqualGreaterEqual8Drop_NXP_3/dropout/random_uniform/RandomUniform:output:0*Drop_NXP_3/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2!
Drop_NXP_3/dropout/GreaterEqualЁ
Drop_NXP_3/dropout/CastCast#Drop_NXP_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџ2
Drop_NXP_3/dropout/CastЇ
Drop_NXP_3/dropout/Mul_1MulDrop_NXP_3/dropout/Mul:z:0Drop_NXP_3/dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Drop_NXP_3/dropout/Mul_1І
targets/MatMul/ReadVariableOpReadVariableOp&targets_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
targets/MatMul/ReadVariableOpЁ
targets/MatMulMatMulDrop_NXP_3/dropout/Mul_1:z:0%targets/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
targets/MatMulЄ
targets/BiasAdd/ReadVariableOpReadVariableOp'targets_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
targets/BiasAdd/ReadVariableOpЁ
targets/BiasAddBiasAddtargets/MatMul:product:0&targets/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
targets/BiasAddp
targets/TanhTanhtargets/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
targets/Tanhт	
IdentityIdentitytargets/Tanh:y:0^BN_1/AssignNewValue^BN_1/AssignNewValue_1%^BN_1/FusedBatchNormV3/ReadVariableOp'^BN_1/FusedBatchNormV3/ReadVariableOp_1^BN_1/ReadVariableOp^BN_1/ReadVariableOp_1^BN_2/AssignNewValue^BN_2/AssignNewValue_1%^BN_2/FusedBatchNormV3/ReadVariableOp'^BN_2/FusedBatchNormV3/ReadVariableOp_1^BN_2/ReadVariableOp^BN_2/ReadVariableOp_1 ^Conv_1-2/BiasAdd/ReadVariableOp^Conv_1-2/Conv2D/ReadVariableOp ^Conv_2-1/BiasAdd/ReadVariableOp^Conv_2-1/Conv2D/ReadVariableOp ^Conv_2-2/BiasAdd/ReadVariableOp^Conv_2-2/Conv2D/ReadVariableOp ^Conv_3-1/BiasAdd/ReadVariableOp^Conv_3-1/Conv2D/ReadVariableOp ^Conv_3-2/BiasAdd/ReadVariableOp^Conv_3-2/Conv2D/ReadVariableOp ^Conv_4-1/BiasAdd/ReadVariableOp^Conv_4-1/Conv2D/ReadVariableOp ^Conv_4-2/BiasAdd/ReadVariableOp^Conv_4-2/Conv2D/ReadVariableOp^Dense_1/BiasAdd/ReadVariableOp^Dense_1/MatMul/ReadVariableOp#^Dense_NXP_2/BiasAdd/ReadVariableOp"^Dense_NXP_2/MatMul/ReadVariableOp#^Dense_NXP_3/BiasAdd/ReadVariableOp"^Dense_NXP_3/MatMul/ReadVariableOp^input/BiasAdd/ReadVariableOp^input/Conv2D/ReadVariableOp^targets/BiasAdd/ReadVariableOp^targets/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:џџџџџџџџџии: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
BN_1/AssignNewValueBN_1/AssignNewValue2.
BN_1/AssignNewValue_1BN_1/AssignNewValue_12L
$BN_1/FusedBatchNormV3/ReadVariableOp$BN_1/FusedBatchNormV3/ReadVariableOp2P
&BN_1/FusedBatchNormV3/ReadVariableOp_1&BN_1/FusedBatchNormV3/ReadVariableOp_12*
BN_1/ReadVariableOpBN_1/ReadVariableOp2.
BN_1/ReadVariableOp_1BN_1/ReadVariableOp_12*
BN_2/AssignNewValueBN_2/AssignNewValue2.
BN_2/AssignNewValue_1BN_2/AssignNewValue_12L
$BN_2/FusedBatchNormV3/ReadVariableOp$BN_2/FusedBatchNormV3/ReadVariableOp2P
&BN_2/FusedBatchNormV3/ReadVariableOp_1&BN_2/FusedBatchNormV3/ReadVariableOp_12*
BN_2/ReadVariableOpBN_2/ReadVariableOp2.
BN_2/ReadVariableOp_1BN_2/ReadVariableOp_12B
Conv_1-2/BiasAdd/ReadVariableOpConv_1-2/BiasAdd/ReadVariableOp2@
Conv_1-2/Conv2D/ReadVariableOpConv_1-2/Conv2D/ReadVariableOp2B
Conv_2-1/BiasAdd/ReadVariableOpConv_2-1/BiasAdd/ReadVariableOp2@
Conv_2-1/Conv2D/ReadVariableOpConv_2-1/Conv2D/ReadVariableOp2B
Conv_2-2/BiasAdd/ReadVariableOpConv_2-2/BiasAdd/ReadVariableOp2@
Conv_2-2/Conv2D/ReadVariableOpConv_2-2/Conv2D/ReadVariableOp2B
Conv_3-1/BiasAdd/ReadVariableOpConv_3-1/BiasAdd/ReadVariableOp2@
Conv_3-1/Conv2D/ReadVariableOpConv_3-1/Conv2D/ReadVariableOp2B
Conv_3-2/BiasAdd/ReadVariableOpConv_3-2/BiasAdd/ReadVariableOp2@
Conv_3-2/Conv2D/ReadVariableOpConv_3-2/Conv2D/ReadVariableOp2B
Conv_4-1/BiasAdd/ReadVariableOpConv_4-1/BiasAdd/ReadVariableOp2@
Conv_4-1/Conv2D/ReadVariableOpConv_4-1/Conv2D/ReadVariableOp2B
Conv_4-2/BiasAdd/ReadVariableOpConv_4-2/BiasAdd/ReadVariableOp2@
Conv_4-2/Conv2D/ReadVariableOpConv_4-2/Conv2D/ReadVariableOp2@
Dense_1/BiasAdd/ReadVariableOpDense_1/BiasAdd/ReadVariableOp2>
Dense_1/MatMul/ReadVariableOpDense_1/MatMul/ReadVariableOp2H
"Dense_NXP_2/BiasAdd/ReadVariableOp"Dense_NXP_2/BiasAdd/ReadVariableOp2F
!Dense_NXP_2/MatMul/ReadVariableOp!Dense_NXP_2/MatMul/ReadVariableOp2H
"Dense_NXP_3/BiasAdd/ReadVariableOp"Dense_NXP_3/BiasAdd/ReadVariableOp2F
!Dense_NXP_3/MatMul/ReadVariableOp!Dense_NXP_3/MatMul/ReadVariableOp2<
input/BiasAdd/ReadVariableOpinput/BiasAdd/ReadVariableOp2:
input/Conv2D/ReadVariableOpinput/Conv2D/ReadVariableOp2@
targets/BiasAdd/ReadVariableOptargets/BiasAdd/ReadVariableOp2>
targets/MatMul/ReadVariableOptargets/MatMul/ReadVariableOp:Y U
1
_output_shapes
:џџџџџџџџџии
 
_user_specified_nameinputs
дc
ъ
G__inference_sequential_1_layer_call_and_return_conditional_losses_34090
input_input%
input_34003:
input_34005:(
conv_1_2_34008:
conv_1_2_34010:(
conv_2_1_34014:
conv_2_1_34016:(
conv_2_2_34019:
conv_2_2_34021:

bn_1_34024:

bn_1_34026:

bn_1_34028:

bn_1_34030:(
conv_3_1_34034: 
conv_3_1_34036: (
conv_3_2_34039:  
conv_3_2_34041: (
conv_4_1_34045:  
conv_4_1_34047: (
conv_4_2_34050:  
conv_4_2_34052: 

bn_2_34055: 

bn_2_34057: 

bn_2_34059: 

bn_2_34061: !
dense_1_34066:
 
dense_1_34068:	%
dense_nxp_2_34072:
 
dense_nxp_2_34074:	%
dense_nxp_3_34078:
 
dense_nxp_3_34080:	 
targets_34084:	
targets_34086:
identityЂBN_1/StatefulPartitionedCallЂBN_2/StatefulPartitionedCallЂ Conv_1-2/StatefulPartitionedCallЂ Conv_2-1/StatefulPartitionedCallЂ Conv_2-2/StatefulPartitionedCallЂ Conv_3-1/StatefulPartitionedCallЂ Conv_3-2/StatefulPartitionedCallЂ Conv_4-1/StatefulPartitionedCallЂ Conv_4-2/StatefulPartitionedCallЂDense_1/StatefulPartitionedCallЂ#Dense_NXP_2/StatefulPartitionedCallЂ#Dense_NXP_3/StatefulPartitionedCallЂinput/StatefulPartitionedCallЂtargets/StatefulPartitionedCall
input/StatefulPartitionedCallStatefulPartitionedCallinput_inputinput_34003input_34005*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџжж*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_input_layer_call_and_return_conditional_losses_330722
input/StatefulPartitionedCallО
 Conv_1-2/StatefulPartitionedCallStatefulPartitionedCall&input/StatefulPartitionedCall:output:0conv_1_2_34008conv_1_2_34010*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџдд*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Conv_1-2_layer_call_and_return_conditional_losses_330892"
 Conv_1-2/StatefulPartitionedCallћ
Pool_1/PartitionedCallPartitionedCall)Conv_1-2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџjj* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_Pool_1_layer_call_and_return_conditional_losses_327602
Pool_1/PartitionedCallЕ
 Conv_2-1/StatefulPartitionedCallStatefulPartitionedCallPool_1/PartitionedCall:output:0conv_2_1_34014conv_2_1_34016*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџhh*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Conv_2-1_layer_call_and_return_conditional_losses_331072"
 Conv_2-1/StatefulPartitionedCallП
 Conv_2-2/StatefulPartitionedCallStatefulPartitionedCall)Conv_2-1/StatefulPartitionedCall:output:0conv_2_2_34019conv_2_2_34021*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџff*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Conv_2-2_layer_call_and_return_conditional_losses_331242"
 Conv_2-2/StatefulPartitionedCallЧ
BN_1/StatefulPartitionedCallStatefulPartitionedCall)Conv_2-2/StatefulPartitionedCall:output:0
bn_1_34024
bn_1_34026
bn_1_34028
bn_1_34030*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџff*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_BN_1_layer_call_and_return_conditional_losses_331472
BN_1/StatefulPartitionedCallї
Pool_2/PartitionedCallPartitionedCall%BN_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ33* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_Pool_2_layer_call_and_return_conditional_losses_328982
Pool_2/PartitionedCallЕ
 Conv_3-1/StatefulPartitionedCallStatefulPartitionedCallPool_2/PartitionedCall:output:0conv_3_1_34034conv_3_1_34036*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ11 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Conv_3-1_layer_call_and_return_conditional_losses_331692"
 Conv_3-1/StatefulPartitionedCallП
 Conv_3-2/StatefulPartitionedCallStatefulPartitionedCall)Conv_3-1/StatefulPartitionedCall:output:0conv_3_2_34039conv_3_2_34041*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ// *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Conv_3-2_layer_call_and_return_conditional_losses_331862"
 Conv_3-2/StatefulPartitionedCallћ
Pool_3/PartitionedCallPartitionedCall)Conv_3-2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_Pool_3_layer_call_and_return_conditional_losses_329102
Pool_3/PartitionedCallЕ
 Conv_4-1/StatefulPartitionedCallStatefulPartitionedCallPool_3/PartitionedCall:output:0conv_4_1_34045conv_4_1_34047*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Conv_4-1_layer_call_and_return_conditional_losses_332042"
 Conv_4-1/StatefulPartitionedCallП
 Conv_4-2/StatefulPartitionedCallStatefulPartitionedCall)Conv_4-1/StatefulPartitionedCall:output:0conv_4_2_34050conv_4_2_34052*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Conv_4-2_layer_call_and_return_conditional_losses_332212"
 Conv_4-2/StatefulPartitionedCallЧ
BN_2/StatefulPartitionedCallStatefulPartitionedCall)Conv_4-2/StatefulPartitionedCall:output:0
bn_2_34055
bn_2_34057
bn_2_34059
bn_2_34061*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_BN_2_layer_call_and_return_conditional_losses_332442
BN_2/StatefulPartitionedCallї
Pool_4/PartitionedCallPartitionedCall%BN_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ		 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_Pool_4_layer_call_and_return_conditional_losses_330482
Pool_4/PartitionedCallѓ
Flatten_1/PartitionedCallPartitionedCallPool_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Flatten_1_layer_call_and_return_conditional_losses_332612
Flatten_1/PartitionedCallЌ
Dense_1/StatefulPartitionedCallStatefulPartitionedCall"Flatten_1/PartitionedCall:output:0dense_1_34066dense_1_34068*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_Dense_1_layer_call_and_return_conditional_losses_332742!
Dense_1/StatefulPartitionedCallѓ
Drop_1/PartitionedCallPartitionedCall(Dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_Drop_1_layer_call_and_return_conditional_losses_332852
Drop_1/PartitionedCallН
#Dense_NXP_2/StatefulPartitionedCallStatefulPartitionedCallDrop_1/PartitionedCall:output:0dense_nxp_2_34072dense_nxp_2_34074*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_Dense_NXP_2_layer_call_and_return_conditional_losses_332982%
#Dense_NXP_2/StatefulPartitionedCall
Drop_NXP_2/PartitionedCallPartitionedCall,Dense_NXP_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_Drop_NXP_2_layer_call_and_return_conditional_losses_333092
Drop_NXP_2/PartitionedCallС
#Dense_NXP_3/StatefulPartitionedCallStatefulPartitionedCall#Drop_NXP_2/PartitionedCall:output:0dense_nxp_3_34078dense_nxp_3_34080*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_Dense_NXP_3_layer_call_and_return_conditional_losses_333222%
#Dense_NXP_3/StatefulPartitionedCall
Drop_NXP_3/PartitionedCallPartitionedCall,Dense_NXP_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_Drop_NXP_3_layer_call_and_return_conditional_losses_333332
Drop_NXP_3/PartitionedCallЌ
targets/StatefulPartitionedCallStatefulPartitionedCall#Drop_NXP_3/PartitionedCall:output:0targets_34084targets_34086*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_targets_layer_call_and_return_conditional_losses_333462!
targets/StatefulPartitionedCallп
IdentityIdentity(targets/StatefulPartitionedCall:output:0^BN_1/StatefulPartitionedCall^BN_2/StatefulPartitionedCall!^Conv_1-2/StatefulPartitionedCall!^Conv_2-1/StatefulPartitionedCall!^Conv_2-2/StatefulPartitionedCall!^Conv_3-1/StatefulPartitionedCall!^Conv_3-2/StatefulPartitionedCall!^Conv_4-1/StatefulPartitionedCall!^Conv_4-2/StatefulPartitionedCall ^Dense_1/StatefulPartitionedCall$^Dense_NXP_2/StatefulPartitionedCall$^Dense_NXP_3/StatefulPartitionedCall^input/StatefulPartitionedCall ^targets/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:џџџџџџџџџии: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
BN_1/StatefulPartitionedCallBN_1/StatefulPartitionedCall2<
BN_2/StatefulPartitionedCallBN_2/StatefulPartitionedCall2D
 Conv_1-2/StatefulPartitionedCall Conv_1-2/StatefulPartitionedCall2D
 Conv_2-1/StatefulPartitionedCall Conv_2-1/StatefulPartitionedCall2D
 Conv_2-2/StatefulPartitionedCall Conv_2-2/StatefulPartitionedCall2D
 Conv_3-1/StatefulPartitionedCall Conv_3-1/StatefulPartitionedCall2D
 Conv_3-2/StatefulPartitionedCall Conv_3-2/StatefulPartitionedCall2D
 Conv_4-1/StatefulPartitionedCall Conv_4-1/StatefulPartitionedCall2D
 Conv_4-2/StatefulPartitionedCall Conv_4-2/StatefulPartitionedCall2B
Dense_1/StatefulPartitionedCallDense_1/StatefulPartitionedCall2J
#Dense_NXP_2/StatefulPartitionedCall#Dense_NXP_2/StatefulPartitionedCall2J
#Dense_NXP_3/StatefulPartitionedCall#Dense_NXP_3/StatefulPartitionedCall2>
input/StatefulPartitionedCallinput/StatefulPartitionedCall2B
targets/StatefulPartitionedCalltargets/StatefulPartitionedCall:^ Z
1
_output_shapes
:џџџџџџџџџии
%
_user_specified_nameinput_input
Ф

(__inference_Conv_3-2_layer_call_fn_34899

inputs!
unknown:  
	unknown_0: 
identityЂStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ// *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Conv_3-2_layer_call_and_return_conditional_losses_331862
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ// 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ11 : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ11 
 
_user_specified_nameinputs
Ц
B
&__inference_Pool_3_layer_call_fn_32916

inputs
identityх
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_Pool_3_layer_call_and_return_conditional_losses_329102
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ђh
е
G__inference_sequential_1_layer_call_and_return_conditional_losses_34180
input_input%
input_34093:
input_34095:(
conv_1_2_34098:
conv_1_2_34100:(
conv_2_1_34104:
conv_2_1_34106:(
conv_2_2_34109:
conv_2_2_34111:

bn_1_34114:

bn_1_34116:

bn_1_34118:

bn_1_34120:(
conv_3_1_34124: 
conv_3_1_34126: (
conv_3_2_34129:  
conv_3_2_34131: (
conv_4_1_34135:  
conv_4_1_34137: (
conv_4_2_34140:  
conv_4_2_34142: 

bn_2_34145: 

bn_2_34147: 

bn_2_34149: 

bn_2_34151: !
dense_1_34156:
 
dense_1_34158:	%
dense_nxp_2_34162:
 
dense_nxp_2_34164:	%
dense_nxp_3_34168:
 
dense_nxp_3_34170:	 
targets_34174:	
targets_34176:
identityЂBN_1/StatefulPartitionedCallЂBN_2/StatefulPartitionedCallЂ Conv_1-2/StatefulPartitionedCallЂ Conv_2-1/StatefulPartitionedCallЂ Conv_2-2/StatefulPartitionedCallЂ Conv_3-1/StatefulPartitionedCallЂ Conv_3-2/StatefulPartitionedCallЂ Conv_4-1/StatefulPartitionedCallЂ Conv_4-2/StatefulPartitionedCallЂDense_1/StatefulPartitionedCallЂ#Dense_NXP_2/StatefulPartitionedCallЂ#Dense_NXP_3/StatefulPartitionedCallЂDrop_1/StatefulPartitionedCallЂ"Drop_NXP_2/StatefulPartitionedCallЂ"Drop_NXP_3/StatefulPartitionedCallЂinput/StatefulPartitionedCallЂtargets/StatefulPartitionedCall
input/StatefulPartitionedCallStatefulPartitionedCallinput_inputinput_34093input_34095*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџжж*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_input_layer_call_and_return_conditional_losses_330722
input/StatefulPartitionedCallО
 Conv_1-2/StatefulPartitionedCallStatefulPartitionedCall&input/StatefulPartitionedCall:output:0conv_1_2_34098conv_1_2_34100*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџдд*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Conv_1-2_layer_call_and_return_conditional_losses_330892"
 Conv_1-2/StatefulPartitionedCallћ
Pool_1/PartitionedCallPartitionedCall)Conv_1-2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџjj* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_Pool_1_layer_call_and_return_conditional_losses_327602
Pool_1/PartitionedCallЕ
 Conv_2-1/StatefulPartitionedCallStatefulPartitionedCallPool_1/PartitionedCall:output:0conv_2_1_34104conv_2_1_34106*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџhh*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Conv_2-1_layer_call_and_return_conditional_losses_331072"
 Conv_2-1/StatefulPartitionedCallП
 Conv_2-2/StatefulPartitionedCallStatefulPartitionedCall)Conv_2-1/StatefulPartitionedCall:output:0conv_2_2_34109conv_2_2_34111*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџff*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Conv_2-2_layer_call_and_return_conditional_losses_331242"
 Conv_2-2/StatefulPartitionedCallХ
BN_1/StatefulPartitionedCallStatefulPartitionedCall)Conv_2-2/StatefulPartitionedCall:output:0
bn_1_34114
bn_1_34116
bn_1_34118
bn_1_34120*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџff*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_BN_1_layer_call_and_return_conditional_losses_336522
BN_1/StatefulPartitionedCallї
Pool_2/PartitionedCallPartitionedCall%BN_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ33* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_Pool_2_layer_call_and_return_conditional_losses_328982
Pool_2/PartitionedCallЕ
 Conv_3-1/StatefulPartitionedCallStatefulPartitionedCallPool_2/PartitionedCall:output:0conv_3_1_34124conv_3_1_34126*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ11 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Conv_3-1_layer_call_and_return_conditional_losses_331692"
 Conv_3-1/StatefulPartitionedCallП
 Conv_3-2/StatefulPartitionedCallStatefulPartitionedCall)Conv_3-1/StatefulPartitionedCall:output:0conv_3_2_34129conv_3_2_34131*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ// *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Conv_3-2_layer_call_and_return_conditional_losses_331862"
 Conv_3-2/StatefulPartitionedCallћ
Pool_3/PartitionedCallPartitionedCall)Conv_3-2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_Pool_3_layer_call_and_return_conditional_losses_329102
Pool_3/PartitionedCallЕ
 Conv_4-1/StatefulPartitionedCallStatefulPartitionedCallPool_3/PartitionedCall:output:0conv_4_1_34135conv_4_1_34137*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Conv_4-1_layer_call_and_return_conditional_losses_332042"
 Conv_4-1/StatefulPartitionedCallП
 Conv_4-2/StatefulPartitionedCallStatefulPartitionedCall)Conv_4-1/StatefulPartitionedCall:output:0conv_4_2_34140conv_4_2_34142*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Conv_4-2_layer_call_and_return_conditional_losses_332212"
 Conv_4-2/StatefulPartitionedCallХ
BN_2/StatefulPartitionedCallStatefulPartitionedCall)Conv_4-2/StatefulPartitionedCall:output:0
bn_2_34145
bn_2_34147
bn_2_34149
bn_2_34151*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_BN_2_layer_call_and_return_conditional_losses_335682
BN_2/StatefulPartitionedCallї
Pool_4/PartitionedCallPartitionedCall%BN_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ		 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_Pool_4_layer_call_and_return_conditional_losses_330482
Pool_4/PartitionedCallѓ
Flatten_1/PartitionedCallPartitionedCallPool_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Flatten_1_layer_call_and_return_conditional_losses_332612
Flatten_1/PartitionedCallЌ
Dense_1/StatefulPartitionedCallStatefulPartitionedCall"Flatten_1/PartitionedCall:output:0dense_1_34156dense_1_34158*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_Dense_1_layer_call_and_return_conditional_losses_332742!
Dense_1/StatefulPartitionedCall
Drop_1/StatefulPartitionedCallStatefulPartitionedCall(Dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_Drop_1_layer_call_and_return_conditional_losses_335162 
Drop_1/StatefulPartitionedCallХ
#Dense_NXP_2/StatefulPartitionedCallStatefulPartitionedCall'Drop_1/StatefulPartitionedCall:output:0dense_nxp_2_34162dense_nxp_2_34164*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_Dense_NXP_2_layer_call_and_return_conditional_losses_332982%
#Dense_NXP_2/StatefulPartitionedCallМ
"Drop_NXP_2/StatefulPartitionedCallStatefulPartitionedCall,Dense_NXP_2/StatefulPartitionedCall:output:0^Drop_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_Drop_NXP_2_layer_call_and_return_conditional_losses_334832$
"Drop_NXP_2/StatefulPartitionedCallЩ
#Dense_NXP_3/StatefulPartitionedCallStatefulPartitionedCall+Drop_NXP_2/StatefulPartitionedCall:output:0dense_nxp_3_34168dense_nxp_3_34170*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_Dense_NXP_3_layer_call_and_return_conditional_losses_333222%
#Dense_NXP_3/StatefulPartitionedCallР
"Drop_NXP_3/StatefulPartitionedCallStatefulPartitionedCall,Dense_NXP_3/StatefulPartitionedCall:output:0#^Drop_NXP_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_Drop_NXP_3_layer_call_and_return_conditional_losses_334502$
"Drop_NXP_3/StatefulPartitionedCallД
targets/StatefulPartitionedCallStatefulPartitionedCall+Drop_NXP_3/StatefulPartitionedCall:output:0targets_34174targets_34176*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_targets_layer_call_and_return_conditional_losses_333462!
targets/StatefulPartitionedCallЪ
IdentityIdentity(targets/StatefulPartitionedCall:output:0^BN_1/StatefulPartitionedCall^BN_2/StatefulPartitionedCall!^Conv_1-2/StatefulPartitionedCall!^Conv_2-1/StatefulPartitionedCall!^Conv_2-2/StatefulPartitionedCall!^Conv_3-1/StatefulPartitionedCall!^Conv_3-2/StatefulPartitionedCall!^Conv_4-1/StatefulPartitionedCall!^Conv_4-2/StatefulPartitionedCall ^Dense_1/StatefulPartitionedCall$^Dense_NXP_2/StatefulPartitionedCall$^Dense_NXP_3/StatefulPartitionedCall^Drop_1/StatefulPartitionedCall#^Drop_NXP_2/StatefulPartitionedCall#^Drop_NXP_3/StatefulPartitionedCall^input/StatefulPartitionedCall ^targets/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:џџџџџџџџџии: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
BN_1/StatefulPartitionedCallBN_1/StatefulPartitionedCall2<
BN_2/StatefulPartitionedCallBN_2/StatefulPartitionedCall2D
 Conv_1-2/StatefulPartitionedCall Conv_1-2/StatefulPartitionedCall2D
 Conv_2-1/StatefulPartitionedCall Conv_2-1/StatefulPartitionedCall2D
 Conv_2-2/StatefulPartitionedCall Conv_2-2/StatefulPartitionedCall2D
 Conv_3-1/StatefulPartitionedCall Conv_3-1/StatefulPartitionedCall2D
 Conv_3-2/StatefulPartitionedCall Conv_3-2/StatefulPartitionedCall2D
 Conv_4-1/StatefulPartitionedCall Conv_4-1/StatefulPartitionedCall2D
 Conv_4-2/StatefulPartitionedCall Conv_4-2/StatefulPartitionedCall2B
Dense_1/StatefulPartitionedCallDense_1/StatefulPartitionedCall2J
#Dense_NXP_2/StatefulPartitionedCall#Dense_NXP_2/StatefulPartitionedCall2J
#Dense_NXP_3/StatefulPartitionedCall#Dense_NXP_3/StatefulPartitionedCall2@
Drop_1/StatefulPartitionedCallDrop_1/StatefulPartitionedCall2H
"Drop_NXP_2/StatefulPartitionedCall"Drop_NXP_2/StatefulPartitionedCall2H
"Drop_NXP_3/StatefulPartitionedCall"Drop_NXP_3/StatefulPartitionedCall2>
input/StatefulPartitionedCallinput/StatefulPartitionedCall2B
targets/StatefulPartitionedCalltargets/StatefulPartitionedCall:^ Z
1
_output_shapes
:џџџџџџџџџии
%
_user_specified_nameinput_input
ђ
_
A__inference_Drop_1_layer_call_and_return_conditional_losses_35120

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:џџџџџџџџџ2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ж

і
B__inference_Dense_1_layer_call_and_return_conditional_losses_33274

inputs2
matmul_readvariableop_resource:
 .
biasadd_readvariableop_resource:	
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
 *
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Х
F
*__inference_Drop_NXP_3_layer_call_fn_35204

inputs
identityЧ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_Drop_NXP_3_layer_call_and_return_conditional_losses_333332
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

љ
@__inference_input_layer_call_and_return_conditional_losses_33072

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpІ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџжж*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:џџџџџџџџџжж2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:џџџџџџџџџжж2
ReluЁ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:џџџџџџџџџжж2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџии: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:џџџџџџџџџии
 
_user_specified_nameinputs
Д
d
E__inference_Drop_NXP_2_layer_call_and_return_conditional_losses_35179

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeЕ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yП
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ц
`
D__inference_Flatten_1_layer_call_and_return_conditional_losses_33261

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ 
  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџ 2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ		 :W S
/
_output_shapes
:џџџџџџџџџ		 
 
_user_specified_nameinputs
Ь

(__inference_Conv_1-2_layer_call_fn_34695

inputs!
unknown:
	unknown_0:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџдд*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Conv_1-2_layer_call_and_return_conditional_losses_330892
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:џџџџџџџџџдд2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџжж: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:џџџџџџџџџжж
 
_user_specified_nameinputs
Ц
B
&__inference_Pool_1_layer_call_fn_32766

inputs
identityх
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_Pool_1_layer_call_and_return_conditional_losses_327602
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Д
d
E__inference_Drop_NXP_3_layer_call_and_return_conditional_losses_33450

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nлЖ?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeЕ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2
dropout/GreaterEqual/yП
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


,__inference_sequential_1_layer_call_fn_34000
input_input!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:$

unknown_11: 

unknown_12: $

unknown_13:  

unknown_14: $

unknown_15:  

unknown_16: $

unknown_17:  

unknown_18: 

unknown_19: 

unknown_20: 

unknown_21: 

unknown_22: 

unknown_23:
 

unknown_24:	

unknown_25:


unknown_26:	

unknown_27:


unknown_28:	

unknown_29:	

unknown_30:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*>
_read_only_resource_inputs 
	
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_338642
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:џџџџџџџџџии: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
1
_output_shapes
:џџџџџџџџџии
%
_user_specified_nameinput_input
­
Ў
?__inference_BN_2_layer_call_and_return_conditional_losses_35038

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : :*
epsilon%o:*
exponential_avg_factor%
з#<2
FusedBatchNormV3Т
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueЮ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs
Х
F
*__inference_Drop_NXP_2_layer_call_fn_35157

inputs
identityЧ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_Drop_NXP_2_layer_call_and_return_conditional_losses_333092
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ц
B
&__inference_Pool_2_layer_call_fn_32904

inputs
identityх
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_Pool_2_layer_call_and_return_conditional_losses_328982
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs


,__inference_sequential_1_layer_call_fn_34395

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:$

unknown_11: 

unknown_12: $

unknown_13:  

unknown_14: $

unknown_15:  

unknown_16: $

unknown_17:  

unknown_18: 

unknown_19: 

unknown_20: 

unknown_21: 

unknown_22: 

unknown_23:
 

unknown_24:	

unknown_25:


unknown_26:	

unknown_27:


unknown_28:	

unknown_29:	

unknown_30:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*>
_read_only_resource_inputs 
	
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_338642
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:џџџџџџџџџии: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:џџџџџџџџџии
 
_user_specified_nameinputs
Ё
]
A__inference_Pool_4_layer_call_and_return_conditional_losses_33048

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
љ

?__inference_BN_2_layer_call_and_return_conditional_losses_35020

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3ь
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs
х
Ў
?__inference_BN_2_layer_call_and_return_conditional_losses_35074

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityЂAssignNewValueЂAssignNewValue_1ЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1и
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџ : : : : :*
epsilon%o:*
exponential_avg_factor%
з#<2
FusedBatchNormV3Т
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValueЮ
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1ў
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
К

њ
F__inference_Dense_NXP_2_layer_call_and_return_conditional_losses_35152

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Б

?__inference_BN_1_layer_call_and_return_conditional_losses_34852

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:џџџџџџџџџff:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3к
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:џџџџџџџџџff2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџff: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:џџџџџџџџџff
 
_user_specified_nameinputs

ќ
C__inference_Conv_4-1_layer_call_and_return_conditional_losses_33204

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOpЄ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Ё
]
A__inference_Pool_3_layer_call_and_return_conditional_losses_32910

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ф

(__inference_Conv_2-2_layer_call_fn_34735

inputs!
unknown:
	unknown_0:
identityЂStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџff*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Conv_2-2_layer_call_and_return_conditional_losses_331242
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџff2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџhh: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџhh
 
_user_specified_nameinputs
Хc
х
G__inference_sequential_1_layer_call_and_return_conditional_losses_33353

inputs%
input_33073:
input_33075:(
conv_1_2_33090:
conv_1_2_33092:(
conv_2_1_33108:
conv_2_1_33110:(
conv_2_2_33125:
conv_2_2_33127:

bn_1_33148:

bn_1_33150:

bn_1_33152:

bn_1_33154:(
conv_3_1_33170: 
conv_3_1_33172: (
conv_3_2_33187:  
conv_3_2_33189: (
conv_4_1_33205:  
conv_4_1_33207: (
conv_4_2_33222:  
conv_4_2_33224: 

bn_2_33245: 

bn_2_33247: 

bn_2_33249: 

bn_2_33251: !
dense_1_33275:
 
dense_1_33277:	%
dense_nxp_2_33299:
 
dense_nxp_2_33301:	%
dense_nxp_3_33323:
 
dense_nxp_3_33325:	 
targets_33347:	
targets_33349:
identityЂBN_1/StatefulPartitionedCallЂBN_2/StatefulPartitionedCallЂ Conv_1-2/StatefulPartitionedCallЂ Conv_2-1/StatefulPartitionedCallЂ Conv_2-2/StatefulPartitionedCallЂ Conv_3-1/StatefulPartitionedCallЂ Conv_3-2/StatefulPartitionedCallЂ Conv_4-1/StatefulPartitionedCallЂ Conv_4-2/StatefulPartitionedCallЂDense_1/StatefulPartitionedCallЂ#Dense_NXP_2/StatefulPartitionedCallЂ#Dense_NXP_3/StatefulPartitionedCallЂinput/StatefulPartitionedCallЂtargets/StatefulPartitionedCall
input/StatefulPartitionedCallStatefulPartitionedCallinputsinput_33073input_33075*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџжж*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_input_layer_call_and_return_conditional_losses_330722
input/StatefulPartitionedCallО
 Conv_1-2/StatefulPartitionedCallStatefulPartitionedCall&input/StatefulPartitionedCall:output:0conv_1_2_33090conv_1_2_33092*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:џџџџџџџџџдд*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Conv_1-2_layer_call_and_return_conditional_losses_330892"
 Conv_1-2/StatefulPartitionedCallћ
Pool_1/PartitionedCallPartitionedCall)Conv_1-2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџjj* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_Pool_1_layer_call_and_return_conditional_losses_327602
Pool_1/PartitionedCallЕ
 Conv_2-1/StatefulPartitionedCallStatefulPartitionedCallPool_1/PartitionedCall:output:0conv_2_1_33108conv_2_1_33110*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџhh*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Conv_2-1_layer_call_and_return_conditional_losses_331072"
 Conv_2-1/StatefulPartitionedCallП
 Conv_2-2/StatefulPartitionedCallStatefulPartitionedCall)Conv_2-1/StatefulPartitionedCall:output:0conv_2_2_33125conv_2_2_33127*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџff*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Conv_2-2_layer_call_and_return_conditional_losses_331242"
 Conv_2-2/StatefulPartitionedCallЧ
BN_1/StatefulPartitionedCallStatefulPartitionedCall)Conv_2-2/StatefulPartitionedCall:output:0
bn_1_33148
bn_1_33150
bn_1_33152
bn_1_33154*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџff*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_BN_1_layer_call_and_return_conditional_losses_331472
BN_1/StatefulPartitionedCallї
Pool_2/PartitionedCallPartitionedCall%BN_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ33* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_Pool_2_layer_call_and_return_conditional_losses_328982
Pool_2/PartitionedCallЕ
 Conv_3-1/StatefulPartitionedCallStatefulPartitionedCallPool_2/PartitionedCall:output:0conv_3_1_33170conv_3_1_33172*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ11 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Conv_3-1_layer_call_and_return_conditional_losses_331692"
 Conv_3-1/StatefulPartitionedCallП
 Conv_3-2/StatefulPartitionedCallStatefulPartitionedCall)Conv_3-1/StatefulPartitionedCall:output:0conv_3_2_33187conv_3_2_33189*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ// *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Conv_3-2_layer_call_and_return_conditional_losses_331862"
 Conv_3-2/StatefulPartitionedCallћ
Pool_3/PartitionedCallPartitionedCall)Conv_3-2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_Pool_3_layer_call_and_return_conditional_losses_329102
Pool_3/PartitionedCallЕ
 Conv_4-1/StatefulPartitionedCallStatefulPartitionedCallPool_3/PartitionedCall:output:0conv_4_1_33205conv_4_1_33207*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Conv_4-1_layer_call_and_return_conditional_losses_332042"
 Conv_4-1/StatefulPartitionedCallП
 Conv_4-2/StatefulPartitionedCallStatefulPartitionedCall)Conv_4-1/StatefulPartitionedCall:output:0conv_4_2_33222conv_4_2_33224*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Conv_4-2_layer_call_and_return_conditional_losses_332212"
 Conv_4-2/StatefulPartitionedCallЧ
BN_2/StatefulPartitionedCallStatefulPartitionedCall)Conv_4-2/StatefulPartitionedCall:output:0
bn_2_33245
bn_2_33247
bn_2_33249
bn_2_33251*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_BN_2_layer_call_and_return_conditional_losses_332442
BN_2/StatefulPartitionedCallї
Pool_4/PartitionedCallPartitionedCall%BN_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ		 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_Pool_4_layer_call_and_return_conditional_losses_330482
Pool_4/PartitionedCallѓ
Flatten_1/PartitionedCallPartitionedCallPool_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Flatten_1_layer_call_and_return_conditional_losses_332612
Flatten_1/PartitionedCallЌ
Dense_1/StatefulPartitionedCallStatefulPartitionedCall"Flatten_1/PartitionedCall:output:0dense_1_33275dense_1_33277*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_Dense_1_layer_call_and_return_conditional_losses_332742!
Dense_1/StatefulPartitionedCallѓ
Drop_1/PartitionedCallPartitionedCall(Dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_Drop_1_layer_call_and_return_conditional_losses_332852
Drop_1/PartitionedCallН
#Dense_NXP_2/StatefulPartitionedCallStatefulPartitionedCallDrop_1/PartitionedCall:output:0dense_nxp_2_33299dense_nxp_2_33301*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_Dense_NXP_2_layer_call_and_return_conditional_losses_332982%
#Dense_NXP_2/StatefulPartitionedCall
Drop_NXP_2/PartitionedCallPartitionedCall,Dense_NXP_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_Drop_NXP_2_layer_call_and_return_conditional_losses_333092
Drop_NXP_2/PartitionedCallС
#Dense_NXP_3/StatefulPartitionedCallStatefulPartitionedCall#Drop_NXP_2/PartitionedCall:output:0dense_nxp_3_33323dense_nxp_3_33325*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_Dense_NXP_3_layer_call_and_return_conditional_losses_333222%
#Dense_NXP_3/StatefulPartitionedCall
Drop_NXP_3/PartitionedCallPartitionedCall,Dense_NXP_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_Drop_NXP_3_layer_call_and_return_conditional_losses_333332
Drop_NXP_3/PartitionedCallЌ
targets/StatefulPartitionedCallStatefulPartitionedCall#Drop_NXP_3/PartitionedCall:output:0targets_33347targets_33349*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_targets_layer_call_and_return_conditional_losses_333462!
targets/StatefulPartitionedCallп
IdentityIdentity(targets/StatefulPartitionedCall:output:0^BN_1/StatefulPartitionedCall^BN_2/StatefulPartitionedCall!^Conv_1-2/StatefulPartitionedCall!^Conv_2-1/StatefulPartitionedCall!^Conv_2-2/StatefulPartitionedCall!^Conv_3-1/StatefulPartitionedCall!^Conv_3-2/StatefulPartitionedCall!^Conv_4-1/StatefulPartitionedCall!^Conv_4-2/StatefulPartitionedCall ^Dense_1/StatefulPartitionedCall$^Dense_NXP_2/StatefulPartitionedCall$^Dense_NXP_3/StatefulPartitionedCall^input/StatefulPartitionedCall ^targets/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:џџџџџџџџџии: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
BN_1/StatefulPartitionedCallBN_1/StatefulPartitionedCall2<
BN_2/StatefulPartitionedCallBN_2/StatefulPartitionedCall2D
 Conv_1-2/StatefulPartitionedCall Conv_1-2/StatefulPartitionedCall2D
 Conv_2-1/StatefulPartitionedCall Conv_2-1/StatefulPartitionedCall2D
 Conv_2-2/StatefulPartitionedCall Conv_2-2/StatefulPartitionedCall2D
 Conv_3-1/StatefulPartitionedCall Conv_3-1/StatefulPartitionedCall2D
 Conv_3-2/StatefulPartitionedCall Conv_3-2/StatefulPartitionedCall2D
 Conv_4-1/StatefulPartitionedCall Conv_4-1/StatefulPartitionedCall2D
 Conv_4-2/StatefulPartitionedCall Conv_4-2/StatefulPartitionedCall2B
Dense_1/StatefulPartitionedCallDense_1/StatefulPartitionedCall2J
#Dense_NXP_2/StatefulPartitionedCall#Dense_NXP_2/StatefulPartitionedCall2J
#Dense_NXP_3/StatefulPartitionedCall#Dense_NXP_3/StatefulPartitionedCall2>
input/StatefulPartitionedCallinput/StatefulPartitionedCall2B
targets/StatefulPartitionedCalltargets/StatefulPartitionedCall:Y U
1
_output_shapes
:џџџџџџџџџии
 
_user_specified_nameinputs
і
c
E__inference_Drop_NXP_2_layer_call_and_return_conditional_losses_33309

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:џџџџџџџџџ2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ў
П
$__inference_BN_2_layer_call_fn_35002

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_BN_2_layer_call_and_return_conditional_losses_335682
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:џџџџџџџџџ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
і
c
E__inference_Drop_NXP_3_layer_call_and_return_conditional_losses_35214

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:џџџџџџџџџ2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ф

(__inference_Conv_4-2_layer_call_fn_34939

inputs!
unknown:  
	unknown_0: 
identityЂStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Conv_4-2_layer_call_and_return_conditional_losses_332212
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
љ

?__inference_BN_1_layer_call_and_return_conditional_losses_34816

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3ь
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
љ

?__inference_BN_1_layer_call_and_return_conditional_losses_32788

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityЂFusedBatchNormV3/ReadVariableOpЂ!FusedBatchNormV3/ReadVariableOp_1ЂReadVariableOpЂReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1Ї
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1м
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3ь
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs


'__inference_targets_layer_call_fn_35235

inputs
unknown:	
	unknown_0:
identityЂStatefulPartitionedCallѕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_targets_layer_call_and_return_conditional_losses_333462
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

ќ
C__inference_Conv_2-2_layer_call_and_return_conditional_losses_33124

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЄ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџff*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџff2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџff2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:џџџџџџџџџff2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџhh: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџhh
 
_user_specified_nameinputs
і
c
E__inference_Drop_NXP_3_layer_call_and_return_conditional_losses_33333

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:џџџџџџџџџ2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ц
B
&__inference_Pool_4_layer_call_fn_33054

inputs
identityх
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_Pool_4_layer_call_and_return_conditional_losses_330482
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

ќ
C__inference_Conv_2-2_layer_call_and_return_conditional_losses_34746

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЄ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџff*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџff2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџff2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:џџџџџџџџџff2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџhh: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџhh
 
_user_specified_nameinputs

ќ
C__inference_Conv_3-1_layer_call_and_return_conditional_losses_33169

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpЄ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ11 *
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ11 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ11 2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:џџџџџџџџџ11 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ33: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ33
 
_user_specified_nameinputs
б
c
*__inference_Drop_NXP_2_layer_call_fn_35162

inputs
identityЂStatefulPartitionedCallп
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_Drop_NXP_2_layer_call_and_return_conditional_losses_334832
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


,__inference_sequential_1_layer_call_fn_34326

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:$

unknown_11: 

unknown_12: $

unknown_13:  

unknown_14: $

unknown_15:  

unknown_16: $

unknown_17:  

unknown_18: 

unknown_19: 

unknown_20: 

unknown_21: 

unknown_22: 

unknown_23:
 

unknown_24:	

unknown_25:


unknown_26:	

unknown_27:


unknown_28:	

unknown_29:	

unknown_30:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*B
_read_only_resource_inputs$
" 	
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_333532
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*p
_input_shapes_
]:џџџџџџџџџии: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:џџџџџџџџџии
 
_user_specified_nameinputs"ЬL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*М
serving_defaultЈ
M
input_input>
serving_default_input_input:0џџџџџџџџџии;
targets0
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:гу
ѕК
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer-6
layer_with_weights-5
layer-7
	layer_with_weights-6
	layer-8

layer-9
layer_with_weights-7
layer-10
layer_with_weights-8
layer-11
layer_with_weights-9
layer-12
layer-13
layer-14
layer_with_weights-10
layer-15
layer-16
layer_with_weights-11
layer-17
layer-18
layer_with_weights-12
layer-19
layer-20
layer_with_weights-13
layer-21
	optimizer

signatures
#_self_saveable_object_factories
regularization_losses
trainable_variables
	variables
	keras_api
ю_default_save_signature
я__call__
+№&call_and_return_all_conditional_losses"чГ
_tf_keras_sequentialЧГ{"name": "sequential_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 216, 216, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_input"}}, {"class_name": "Conv2D", "config": {"name": "input", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 216, 216, 3]}, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "Conv_1-2", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "Pool_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "Conv_2-1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "Conv_2-2", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "BN_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "Pool_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "Conv_3-1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "Conv_3-2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "Pool_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "Conv_4-1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "Conv_4-2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "BN_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "Pool_4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "Flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "Dense_1", "trainable": true, "dtype": "float32", "units": 1024, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "Drop_1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "Dense_NXP_2", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "Drop_NXP_2", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "Dense_NXP_3", "trainable": true, "dtype": "float32", "units": 128, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "Drop_NXP_3", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "targets", "trainable": true, "dtype": "float32", "units": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 55, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}, "shared_object_id": 56}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 216, 216, 3]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 216, 216, 3]}, "float32", "input_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 216, 216, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_input"}, "shared_object_id": 0}, {"class_name": "Conv2D", "config": {"name": "input", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 216, 216, 3]}, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3}, {"class_name": "Conv2D", "config": {"name": "Conv_1-2", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 6}, {"class_name": "MaxPooling2D", "config": {"name": "Pool_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 7}, {"class_name": "Conv2D", "config": {"name": "Conv_2-1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 10}, {"class_name": "Conv2D", "config": {"name": "Conv_2-2", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 13}, {"class_name": "BatchNormalization", "config": {"name": "BN_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 15}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 17}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 18}, {"class_name": "MaxPooling2D", "config": {"name": "Pool_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 19}, {"class_name": "Conv2D", "config": {"name": "Conv_3-1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 20}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 21}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 22}, {"class_name": "Conv2D", "config": {"name": "Conv_3-2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 23}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 24}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 25}, {"class_name": "MaxPooling2D", "config": {"name": "Pool_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 26}, {"class_name": "Conv2D", "config": {"name": "Conv_4-1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 27}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 28}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 29}, {"class_name": "Conv2D", "config": {"name": "Conv_4-2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 30}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 31}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 32}, {"class_name": "BatchNormalization", "config": {"name": "BN_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 33}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 34}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 35}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 36}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 37}, {"class_name": "MaxPooling2D", "config": {"name": "Pool_4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 38}, {"class_name": "Flatten", "config": {"name": "Flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 39}, {"class_name": "Dense", "config": {"name": "Dense_1", "trainable": true, "dtype": "float32", "units": 1024, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 40}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 41}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 42}, {"class_name": "Dropout", "config": {"name": "Drop_1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 43}, {"class_name": "Dense", "config": {"name": "Dense_NXP_2", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 44}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 45}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 46}, {"class_name": "Dropout", "config": {"name": "Drop_NXP_2", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 47}, {"class_name": "Dense", "config": {"name": "Dense_NXP_3", "trainable": true, "dtype": "float32", "units": 128, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 48}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 49}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 50}, {"class_name": "Dropout", "config": {"name": "Drop_NXP_3", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "shared_object_id": 51}, {"class_name": "Dense", "config": {"name": "targets", "trainable": true, "dtype": "float32", "units": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 52}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 53}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 54}]}}, "training_config": {"loss": "mean_squared_error", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "mae", "dtype": "float32", "fn": "mean_absolute_error"}, "shared_object_id": 57}, {"class_name": "RootMeanSquaredError", "config": {"name": "root_mean_squared_error", "dtype": "float32"}, "shared_object_id": 58}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
і

kernel
bias
# _self_saveable_object_factories
!regularization_losses
"trainable_variables
#	variables
$	keras_api
ё__call__
+ђ&call_and_return_all_conditional_losses"Њ

_tf_keras_layer
{"name": "input", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 216, 216, 3]}, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "input", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 216, 216, 3]}, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}, "shared_object_id": 56}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 216, 216, 3]}}
љ


%kernel
&bias
#'_self_saveable_object_factories
(regularization_losses
)trainable_variables
*	variables
+	keras_api
ѓ__call__
+є&call_and_return_all_conditional_losses"­	
_tf_keras_layer	{"name": "Conv_1-2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "Conv_1-2", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}, "shared_object_id": 59}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 214, 214, 16]}}
У
#,_self_saveable_object_factories
-regularization_losses
.trainable_variables
/	variables
0	keras_api
ѕ__call__
+і&call_and_return_all_conditional_losses"
_tf_keras_layerѓ{"name": "Pool_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "Pool_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 7, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 60}}
њ


1kernel
2bias
#3_self_saveable_object_factories
4regularization_losses
5trainable_variables
6	variables
7	keras_api
ї__call__
+ј&call_and_return_all_conditional_losses"Ў	
_tf_keras_layer	{"name": "Conv_2-1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "Conv_2-1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 10, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}, "shared_object_id": 61}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 106, 106, 16]}}
ќ


8kernel
9bias
#:_self_saveable_object_factories
;regularization_losses
<trainable_variables
=	variables
>	keras_api
љ__call__
+њ&call_and_return_all_conditional_losses"А	
_tf_keras_layer	{"name": "Conv_2-2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "Conv_2-2", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 13, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}, "shared_object_id": 62}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 104, 104, 16]}}
б

?axis
	@gamma
Abeta
Bmoving_mean
Cmoving_variance
#D_self_saveable_object_factories
Eregularization_losses
Ftrainable_variables
G	variables
H	keras_api
ћ__call__
+ќ&call_and_return_all_conditional_losses"ж
_tf_keras_layerМ{"name": "BN_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "BN_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 15}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 17}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 18, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 16}}, "shared_object_id": 63}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 102, 102, 16]}}
Ф
#I_self_saveable_object_factories
Jregularization_losses
Ktrainable_variables
L	variables
M	keras_api
§__call__
+ў&call_and_return_all_conditional_losses"
_tf_keras_layerє{"name": "Pool_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "Pool_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 19, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 64}}
њ


Nkernel
Obias
#P_self_saveable_object_factories
Qregularization_losses
Rtrainable_variables
S	variables
T	keras_api
џ__call__
+&call_and_return_all_conditional_losses"Ў	
_tf_keras_layer	{"name": "Conv_3-1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "Conv_3-1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 20}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 21}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 22, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}, "shared_object_id": 65}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 51, 51, 16]}}
њ


Ukernel
Vbias
#W_self_saveable_object_factories
Xregularization_losses
Ytrainable_variables
Z	variables
[	keras_api
__call__
+&call_and_return_all_conditional_losses"Ў	
_tf_keras_layer	{"name": "Conv_3-2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "Conv_3-2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 23}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 24}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 25, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}, "shared_object_id": 66}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 49, 49, 32]}}
Ф
#\_self_saveable_object_factories
]regularization_losses
^trainable_variables
_	variables
`	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layerє{"name": "Pool_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "Pool_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 26, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 67}}
њ


akernel
bbias
#c_self_saveable_object_factories
dregularization_losses
etrainable_variables
f	variables
g	keras_api
__call__
+&call_and_return_all_conditional_losses"Ў	
_tf_keras_layer	{"name": "Conv_4-1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "Conv_4-1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 27}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 28}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 29, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}, "shared_object_id": 68}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 23, 23, 32]}}
њ


hkernel
ibias
#j_self_saveable_object_factories
kregularization_losses
ltrainable_variables
m	variables
n	keras_api
__call__
+&call_and_return_all_conditional_losses"Ў	
_tf_keras_layer	{"name": "Conv_4-2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "Conv_4-2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 30}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 31}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 32, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}, "shared_object_id": 69}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 21, 21, 32]}}
Я

oaxis
	pgamma
qbeta
rmoving_mean
smoving_variance
#t_self_saveable_object_factories
uregularization_losses
vtrainable_variables
w	variables
x	keras_api
__call__
+&call_and_return_all_conditional_losses"д
_tf_keras_layerК{"name": "BN_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "BN_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 33}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 34}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 35}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 36}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "shared_object_id": 37, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}, "shared_object_id": 70}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 19, 19, 32]}}
Ф
#y_self_saveable_object_factories
zregularization_losses
{trainable_variables
|	variables
}	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layerє{"name": "Pool_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "Pool_4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 38, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 71}}
Р
#~_self_saveable_object_factories
regularization_losses
trainable_variables
	variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layerэ{"name": "Flatten_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "Flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 39, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 72}}
	
kernel
	bias
$_self_saveable_object_factories
regularization_losses
trainable_variables
	variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"Б
_tf_keras_layer{"name": "Dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "Dense_1", "trainable": true, "dtype": "float32", "units": 1024, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 40}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 41}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 42, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2592}}, "shared_object_id": 73}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2592]}}
Ѓ
$_self_saveable_object_factories
regularization_losses
trainable_variables
	variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"ш
_tf_keras_layerЮ{"name": "Drop_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "Drop_1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 43}
	
kernel
	bias
$_self_saveable_object_factories
regularization_losses
trainable_variables
	variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"И
_tf_keras_layer{"name": "Dense_NXP_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "Dense_NXP_2", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 44}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 45}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 46, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1024}}, "shared_object_id": 74}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1024]}}
Ћ
$_self_saveable_object_factories
regularization_losses
trainable_variables
	variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"№
_tf_keras_layerж{"name": "Drop_NXP_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "Drop_NXP_2", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "shared_object_id": 47}
	
kernel
	bias
$_self_saveable_object_factories
regularization_losses
trainable_variables
 	variables
Ё	keras_api
__call__
+&call_and_return_all_conditional_losses"Ж
_tf_keras_layer{"name": "Dense_NXP_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "Dense_NXP_3", "trainable": true, "dtype": "float32", "units": 128, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 48}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 49}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 50, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}, "shared_object_id": 75}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
Ћ
$Ђ_self_saveable_object_factories
Ѓregularization_losses
Єtrainable_variables
Ѕ	variables
І	keras_api
__call__
+&call_and_return_all_conditional_losses"№
_tf_keras_layerж{"name": "Drop_NXP_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "Drop_NXP_3", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "shared_object_id": 51}
џ
Їkernel
	Јbias
$Љ_self_saveable_object_factories
Њregularization_losses
Ћtrainable_variables
Ќ	variables
­	keras_api
__call__
+&call_and_return_all_conditional_losses"Ќ
_tf_keras_layer{"name": "targets", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "targets", "trainable": true, "dtype": "float32", "units": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 52}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 53}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 54, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}, "shared_object_id": 76}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}

	Ўiter
Џbeta_1
Аbeta_2

Бdecay
Вlearning_ratemЖmЗ%mИ&mЙ1mК2mЛ8mМ9mН@mОAmПNmРOmСUmТVmУamФbmХhmЦimЧpmШqmЩ	mЪ	mЫ	mЬ	mЭ	mЮ	mЯ	Їmа	Јmбvвvг%vд&vе1vж2vз8vи9vй@vкAvлNvмOvнUvоVvпavрbvсhvтivуpvфqvх	vц	vч	vш	vщ	vъ	vы	Їvь	Јvэ"
	optimizer
-
serving_default"
signature_map
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
ў
0
1
%2
&3
14
25
86
97
@8
A9
N10
O11
U12
V13
a14
b15
h16
i17
p18
q19
20
21
22
23
24
25
Ї26
Ј27"
trackable_list_wrapper

0
1
%2
&3
14
25
86
97
@8
A9
B10
C11
N12
O13
U14
V15
a16
b17
h18
i19
p20
q21
r22
s23
24
25
26
27
28
29
Ї30
Ј31"
trackable_list_wrapper
г
Гlayers
regularization_losses
trainable_variables
 Дlayer_regularization_losses
Еlayer_metrics
Жmetrics
	variables
Зnon_trainable_variables
я__call__
ю_default_save_signature
+№&call_and_return_all_conditional_losses
'№"call_and_return_conditional_losses"
_generic_user_object
&:$2input/kernel
:2
input/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
Е
Иlayers
!regularization_losses
"trainable_variables
 Йlayer_regularization_losses
Кlayer_metrics
Лmetrics
#	variables
Мnon_trainable_variables
ё__call__
+ђ&call_and_return_all_conditional_losses
'ђ"call_and_return_conditional_losses"
_generic_user_object
):'2Conv_1-2/kernel
:2Conv_1-2/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
Е
Нlayers
(regularization_losses
)trainable_variables
 Оlayer_regularization_losses
Пlayer_metrics
Рmetrics
*	variables
Сnon_trainable_variables
ѓ__call__
+є&call_and_return_all_conditional_losses
'є"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
Тlayers
-regularization_losses
.trainable_variables
 Уlayer_regularization_losses
Фlayer_metrics
Хmetrics
/	variables
Цnon_trainable_variables
ѕ__call__
+і&call_and_return_all_conditional_losses
'і"call_and_return_conditional_losses"
_generic_user_object
):'2Conv_2-1/kernel
:2Conv_2-1/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
Е
Чlayers
4regularization_losses
5trainable_variables
 Шlayer_regularization_losses
Щlayer_metrics
Ъmetrics
6	variables
Ыnon_trainable_variables
ї__call__
+ј&call_and_return_all_conditional_losses
'ј"call_and_return_conditional_losses"
_generic_user_object
):'2Conv_2-2/kernel
:2Conv_2-2/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
Е
Ьlayers
;regularization_losses
<trainable_variables
 Эlayer_regularization_losses
Юlayer_metrics
Яmetrics
=	variables
аnon_trainable_variables
љ__call__
+њ&call_and_return_all_conditional_losses
'њ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:2
BN_1/gamma
:2	BN_1/beta
 : (2BN_1/moving_mean
$:" (2BN_1/moving_variance
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
<
@0
A1
B2
C3"
trackable_list_wrapper
Е
бlayers
Eregularization_losses
Ftrainable_variables
 вlayer_regularization_losses
гlayer_metrics
дmetrics
G	variables
еnon_trainable_variables
ћ__call__
+ќ&call_and_return_all_conditional_losses
'ќ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
жlayers
Jregularization_losses
Ktrainable_variables
 зlayer_regularization_losses
иlayer_metrics
йmetrics
L	variables
кnon_trainable_variables
§__call__
+ў&call_and_return_all_conditional_losses
'ў"call_and_return_conditional_losses"
_generic_user_object
):' 2Conv_3-1/kernel
: 2Conv_3-1/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
N0
O1"
trackable_list_wrapper
.
N0
O1"
trackable_list_wrapper
Е
лlayers
Qregularization_losses
Rtrainable_variables
 мlayer_regularization_losses
нlayer_metrics
оmetrics
S	variables
пnon_trainable_variables
џ__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
):'  2Conv_3-2/kernel
: 2Conv_3-2/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
Е
рlayers
Xregularization_losses
Ytrainable_variables
 сlayer_regularization_losses
тlayer_metrics
уmetrics
Z	variables
фnon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
хlayers
]regularization_losses
^trainable_variables
 цlayer_regularization_losses
чlayer_metrics
шmetrics
_	variables
щnon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
):'  2Conv_4-1/kernel
: 2Conv_4-1/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
a0
b1"
trackable_list_wrapper
.
a0
b1"
trackable_list_wrapper
Е
ъlayers
dregularization_losses
etrainable_variables
 ыlayer_regularization_losses
ьlayer_metrics
эmetrics
f	variables
юnon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
):'  2Conv_4-2/kernel
: 2Conv_4-2/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
h0
i1"
trackable_list_wrapper
.
h0
i1"
trackable_list_wrapper
Е
яlayers
kregularization_losses
ltrainable_variables
 №layer_regularization_losses
ёlayer_metrics
ђmetrics
m	variables
ѓnon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
: 2
BN_2/gamma
: 2	BN_2/beta
 :  (2BN_2/moving_mean
$:"  (2BN_2/moving_variance
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
p0
q1"
trackable_list_wrapper
<
p0
q1
r2
s3"
trackable_list_wrapper
Е
єlayers
uregularization_losses
vtrainable_variables
 ѕlayer_regularization_losses
іlayer_metrics
їmetrics
w	variables
јnon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
љlayers
zregularization_losses
{trainable_variables
 њlayer_regularization_losses
ћlayer_metrics
ќmetrics
|	variables
§non_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
З
ўlayers
regularization_losses
trainable_variables
 џlayer_regularization_losses
layer_metrics
metrics
	variables
non_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
": 
 2Dense_1/kernel
:2Dense_1/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
И
layers
regularization_losses
trainable_variables
 layer_regularization_losses
layer_metrics
metrics
	variables
non_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
layers
regularization_losses
trainable_variables
 layer_regularization_losses
layer_metrics
metrics
	variables
non_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
&:$
2Dense_NXP_2/kernel
:2Dense_NXP_2/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
И
layers
regularization_losses
trainable_variables
 layer_regularization_losses
layer_metrics
metrics
	variables
non_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
layers
regularization_losses
trainable_variables
 layer_regularization_losses
layer_metrics
metrics
	variables
non_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
&:$
2Dense_NXP_3/kernel
:2Dense_NXP_3/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
И
layers
regularization_losses
trainable_variables
 layer_regularization_losses
layer_metrics
metrics
 	variables
non_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
layers
Ѓregularization_losses
Єtrainable_variables
 layer_regularization_losses
layer_metrics
metrics
Ѕ	variables
 non_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
!:	2targets/kernel
:2targets/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
Ї0
Ј1"
trackable_list_wrapper
0
Ї0
Ј1"
trackable_list_wrapper
И
Ёlayers
Њregularization_losses
Ћtrainable_variables
 Ђlayer_regularization_losses
Ѓlayer_metrics
Єmetrics
Ќ	variables
Ѕnon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
Ц
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
8
І0
Ї1
Ј2"
trackable_list_wrapper
<
B0
C1
r2
s3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
r0
s1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
и

Љtotal

Њcount
Ћ	variables
Ќ	keras_api"
_tf_keras_metric{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 77}


­total

Ўcount
Џ
_fn_kwargs
А	variables
Б	keras_api"Х
_tf_keras_metricЊ{"class_name": "MeanMetricWrapper", "name": "mae", "dtype": "float32", "config": {"name": "mae", "dtype": "float32", "fn": "mean_absolute_error"}, "shared_object_id": 57}


Вtotal

Гcount
Д	variables
Е	keras_api"г
_tf_keras_metricИ{"class_name": "RootMeanSquaredError", "name": "root_mean_squared_error", "dtype": "float32", "config": {"name": "root_mean_squared_error", "dtype": "float32"}, "shared_object_id": 58}
:  (2total
:  (2count
0
Љ0
Њ1"
trackable_list_wrapper
.
Ћ	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
­0
Ў1"
trackable_list_wrapper
.
А	variables"
_generic_user_object
:  (2total
:  (2count
0
В0
Г1"
trackable_list_wrapper
.
Д	variables"
_generic_user_object
+:)2Adam/input/kernel/m
:2Adam/input/bias/m
.:,2Adam/Conv_1-2/kernel/m
 :2Adam/Conv_1-2/bias/m
.:,2Adam/Conv_2-1/kernel/m
 :2Adam/Conv_2-1/bias/m
.:,2Adam/Conv_2-2/kernel/m
 :2Adam/Conv_2-2/bias/m
:2Adam/BN_1/gamma/m
:2Adam/BN_1/beta/m
.:, 2Adam/Conv_3-1/kernel/m
 : 2Adam/Conv_3-1/bias/m
.:,  2Adam/Conv_3-2/kernel/m
 : 2Adam/Conv_3-2/bias/m
.:,  2Adam/Conv_4-1/kernel/m
 : 2Adam/Conv_4-1/bias/m
.:,  2Adam/Conv_4-2/kernel/m
 : 2Adam/Conv_4-2/bias/m
: 2Adam/BN_2/gamma/m
: 2Adam/BN_2/beta/m
':%
 2Adam/Dense_1/kernel/m
 :2Adam/Dense_1/bias/m
+:)
2Adam/Dense_NXP_2/kernel/m
$:"2Adam/Dense_NXP_2/bias/m
+:)
2Adam/Dense_NXP_3/kernel/m
$:"2Adam/Dense_NXP_3/bias/m
&:$	2Adam/targets/kernel/m
:2Adam/targets/bias/m
+:)2Adam/input/kernel/v
:2Adam/input/bias/v
.:,2Adam/Conv_1-2/kernel/v
 :2Adam/Conv_1-2/bias/v
.:,2Adam/Conv_2-1/kernel/v
 :2Adam/Conv_2-1/bias/v
.:,2Adam/Conv_2-2/kernel/v
 :2Adam/Conv_2-2/bias/v
:2Adam/BN_1/gamma/v
:2Adam/BN_1/beta/v
.:, 2Adam/Conv_3-1/kernel/v
 : 2Adam/Conv_3-1/bias/v
.:,  2Adam/Conv_3-2/kernel/v
 : 2Adam/Conv_3-2/bias/v
.:,  2Adam/Conv_4-1/kernel/v
 : 2Adam/Conv_4-1/bias/v
.:,  2Adam/Conv_4-2/kernel/v
 : 2Adam/Conv_4-2/bias/v
: 2Adam/BN_2/gamma/v
: 2Adam/BN_2/beta/v
':%
 2Adam/Dense_1/kernel/v
 :2Adam/Dense_1/bias/v
+:)
2Adam/Dense_NXP_2/kernel/v
$:"2Adam/Dense_NXP_2/bias/v
+:)
2Adam/Dense_NXP_3/kernel/v
$:"2Adam/Dense_NXP_3/bias/v
&:$	2Adam/targets/kernel/v
:2Adam/targets/bias/v
ь2щ
 __inference__wrapped_model_32754Ф
В
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *4Ђ1
/,
input_inputџџџџџџџџџии
ў2ћ
,__inference_sequential_1_layer_call_fn_33420
,__inference_sequential_1_layer_call_fn_34326
,__inference_sequential_1_layer_call_fn_34395
,__inference_sequential_1_layer_call_fn_34000Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ъ2ч
G__inference_sequential_1_layer_call_and_return_conditional_losses_34520
G__inference_sequential_1_layer_call_and_return_conditional_losses_34666
G__inference_sequential_1_layer_call_and_return_conditional_losses_34090
G__inference_sequential_1_layer_call_and_return_conditional_losses_34180Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Я2Ь
%__inference_input_layer_call_fn_34675Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ъ2ч
@__inference_input_layer_call_and_return_conditional_losses_34686Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
в2Я
(__inference_Conv_1-2_layer_call_fn_34695Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
э2ъ
C__inference_Conv_1-2_layer_call_and_return_conditional_losses_34706Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
&__inference_Pool_1_layer_call_fn_32766р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Љ2І
A__inference_Pool_1_layer_call_and_return_conditional_losses_32760р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
в2Я
(__inference_Conv_2-1_layer_call_fn_34715Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
э2ъ
C__inference_Conv_2-1_layer_call_and_return_conditional_losses_34726Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
в2Я
(__inference_Conv_2-2_layer_call_fn_34735Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
э2ъ
C__inference_Conv_2-2_layer_call_and_return_conditional_losses_34746Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
в2Я
$__inference_BN_1_layer_call_fn_34759
$__inference_BN_1_layer_call_fn_34772
$__inference_BN_1_layer_call_fn_34785
$__inference_BN_1_layer_call_fn_34798Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
О2Л
?__inference_BN_1_layer_call_and_return_conditional_losses_34816
?__inference_BN_1_layer_call_and_return_conditional_losses_34834
?__inference_BN_1_layer_call_and_return_conditional_losses_34852
?__inference_BN_1_layer_call_and_return_conditional_losses_34870Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
&__inference_Pool_2_layer_call_fn_32904р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Љ2І
A__inference_Pool_2_layer_call_and_return_conditional_losses_32898р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
в2Я
(__inference_Conv_3-1_layer_call_fn_34879Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
э2ъ
C__inference_Conv_3-1_layer_call_and_return_conditional_losses_34890Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
в2Я
(__inference_Conv_3-2_layer_call_fn_34899Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
э2ъ
C__inference_Conv_3-2_layer_call_and_return_conditional_losses_34910Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
&__inference_Pool_3_layer_call_fn_32916р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Љ2І
A__inference_Pool_3_layer_call_and_return_conditional_losses_32910р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
в2Я
(__inference_Conv_4-1_layer_call_fn_34919Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
э2ъ
C__inference_Conv_4-1_layer_call_and_return_conditional_losses_34930Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
в2Я
(__inference_Conv_4-2_layer_call_fn_34939Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
э2ъ
C__inference_Conv_4-2_layer_call_and_return_conditional_losses_34950Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
в2Я
$__inference_BN_2_layer_call_fn_34963
$__inference_BN_2_layer_call_fn_34976
$__inference_BN_2_layer_call_fn_34989
$__inference_BN_2_layer_call_fn_35002Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
О2Л
?__inference_BN_2_layer_call_and_return_conditional_losses_35020
?__inference_BN_2_layer_call_and_return_conditional_losses_35038
?__inference_BN_2_layer_call_and_return_conditional_losses_35056
?__inference_BN_2_layer_call_and_return_conditional_losses_35074Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
&__inference_Pool_4_layer_call_fn_33054р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Љ2І
A__inference_Pool_4_layer_call_and_return_conditional_losses_33048р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
г2а
)__inference_Flatten_1_layer_call_fn_35079Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ю2ы
D__inference_Flatten_1_layer_call_and_return_conditional_losses_35085Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
б2Ю
'__inference_Dense_1_layer_call_fn_35094Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ь2щ
B__inference_Dense_1_layer_call_and_return_conditional_losses_35105Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
&__inference_Drop_1_layer_call_fn_35110
&__inference_Drop_1_layer_call_fn_35115Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Р2Н
A__inference_Drop_1_layer_call_and_return_conditional_losses_35120
A__inference_Drop_1_layer_call_and_return_conditional_losses_35132Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
е2в
+__inference_Dense_NXP_2_layer_call_fn_35141Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
№2э
F__inference_Dense_NXP_2_layer_call_and_return_conditional_losses_35152Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
*__inference_Drop_NXP_2_layer_call_fn_35157
*__inference_Drop_NXP_2_layer_call_fn_35162Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ш2Х
E__inference_Drop_NXP_2_layer_call_and_return_conditional_losses_35167
E__inference_Drop_NXP_2_layer_call_and_return_conditional_losses_35179Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
е2в
+__inference_Dense_NXP_3_layer_call_fn_35188Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
№2э
F__inference_Dense_NXP_3_layer_call_and_return_conditional_losses_35199Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
*__inference_Drop_NXP_3_layer_call_fn_35204
*__inference_Drop_NXP_3_layer_call_fn_35209Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ш2Х
E__inference_Drop_NXP_3_layer_call_and_return_conditional_losses_35214
E__inference_Drop_NXP_3_layer_call_and_return_conditional_losses_35226Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
б2Ю
'__inference_targets_layer_call_fn_35235Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ь2щ
B__inference_targets_layer_call_and_return_conditional_losses_35246Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЮBЫ
#__inference_signature_wrapper_34257input_input"
В
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 к
?__inference_BN_1_layer_call_and_return_conditional_losses_34816@ABCMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 к
?__inference_BN_1_layer_call_and_return_conditional_losses_34834@ABCMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Е
?__inference_BN_1_layer_call_and_return_conditional_losses_34852r@ABC;Ђ8
1Ђ.
(%
inputsџџџџџџџџџff
p 
Њ "-Ђ*
# 
0џџџџџџџџџff
 Е
?__inference_BN_1_layer_call_and_return_conditional_losses_34870r@ABC;Ђ8
1Ђ.
(%
inputsџџџџџџџџџff
p
Њ "-Ђ*
# 
0џџџџџџџџџff
 В
$__inference_BN_1_layer_call_fn_34759@ABCMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџВ
$__inference_BN_1_layer_call_fn_34772@ABCMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
$__inference_BN_1_layer_call_fn_34785e@ABC;Ђ8
1Ђ.
(%
inputsџџџџџџџџџff
p 
Њ " џџџџџџџџџff
$__inference_BN_1_layer_call_fn_34798e@ABC;Ђ8
1Ђ.
(%
inputsџџџџџџџџџff
p
Њ " џџџџџџџџџffк
?__inference_BN_2_layer_call_and_return_conditional_losses_35020pqrsMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 к
?__inference_BN_2_layer_call_and_return_conditional_losses_35038pqrsMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 Е
?__inference_BN_2_layer_call_and_return_conditional_losses_35056rpqrs;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ 
p 
Њ "-Ђ*
# 
0џџџџџџџџџ 
 Е
?__inference_BN_2_layer_call_and_return_conditional_losses_35074rpqrs;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ 
p
Њ "-Ђ*
# 
0џџџџџџџџџ 
 В
$__inference_BN_2_layer_call_fn_34963pqrsMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ В
$__inference_BN_2_layer_call_fn_34976pqrsMЂJ
CЂ@
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
$__inference_BN_2_layer_call_fn_34989epqrs;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ 
p 
Њ " џџџџџџџџџ 
$__inference_BN_2_layer_call_fn_35002epqrs;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ 
p
Њ " џџџџџџџџџ З
C__inference_Conv_1-2_layer_call_and_return_conditional_losses_34706p%&9Ђ6
/Ђ,
*'
inputsџџџџџџџџџжж
Њ "/Ђ,
%"
0џџџџџџџџџдд
 
(__inference_Conv_1-2_layer_call_fn_34695c%&9Ђ6
/Ђ,
*'
inputsџџџџџџџџџжж
Њ ""џџџџџџџџџддГ
C__inference_Conv_2-1_layer_call_and_return_conditional_losses_34726l127Ђ4
-Ђ*
(%
inputsџџџџџџџџџjj
Њ "-Ђ*
# 
0џџџџџџџџџhh
 
(__inference_Conv_2-1_layer_call_fn_34715_127Ђ4
-Ђ*
(%
inputsџџџџџџџџџjj
Њ " џџџџџџџџџhhГ
C__inference_Conv_2-2_layer_call_and_return_conditional_losses_34746l897Ђ4
-Ђ*
(%
inputsџџџџџџџџџhh
Њ "-Ђ*
# 
0џџџџџџџџџff
 
(__inference_Conv_2-2_layer_call_fn_34735_897Ђ4
-Ђ*
(%
inputsџџџџџџџџџhh
Њ " џџџџџџџџџffГ
C__inference_Conv_3-1_layer_call_and_return_conditional_losses_34890lNO7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ33
Њ "-Ђ*
# 
0џџџџџџџџџ11 
 
(__inference_Conv_3-1_layer_call_fn_34879_NO7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ33
Њ " џџџџџџџџџ11 Г
C__inference_Conv_3-2_layer_call_and_return_conditional_losses_34910lUV7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ11 
Њ "-Ђ*
# 
0џџџџџџџџџ// 
 
(__inference_Conv_3-2_layer_call_fn_34899_UV7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ11 
Њ " џџџџџџџџџ// Г
C__inference_Conv_4-1_layer_call_and_return_conditional_losses_34930lab7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ 
Њ "-Ђ*
# 
0џџџџџџџџџ 
 
(__inference_Conv_4-1_layer_call_fn_34919_ab7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ 
Њ " џџџџџџџџџ Г
C__inference_Conv_4-2_layer_call_and_return_conditional_losses_34950lhi7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ 
Њ "-Ђ*
# 
0џџџџџџџџџ 
 
(__inference_Conv_4-2_layer_call_fn_34939_hi7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ 
Њ " џџџџџџџџџ І
B__inference_Dense_1_layer_call_and_return_conditional_losses_35105`0Ђ-
&Ђ#
!
inputsџџџџџџџџџ 
Њ "&Ђ#

0џџџџџџџџџ
 ~
'__inference_Dense_1_layer_call_fn_35094S0Ђ-
&Ђ#
!
inputsџџџџџџџџџ 
Њ "џџџџџџџџџЊ
F__inference_Dense_NXP_2_layer_call_and_return_conditional_losses_35152`0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "&Ђ#

0џџџџџџџџџ
 
+__inference_Dense_NXP_2_layer_call_fn_35141S0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "џџџџџџџџџЊ
F__inference_Dense_NXP_3_layer_call_and_return_conditional_losses_35199`0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "&Ђ#

0џџџџџџџџџ
 
+__inference_Dense_NXP_3_layer_call_fn_35188S0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "џџџџџџџџџЃ
A__inference_Drop_1_layer_call_and_return_conditional_losses_35120^4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p 
Њ "&Ђ#

0џџџџџџџџџ
 Ѓ
A__inference_Drop_1_layer_call_and_return_conditional_losses_35132^4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p
Њ "&Ђ#

0џџџџџџџџџ
 {
&__inference_Drop_1_layer_call_fn_35110Q4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p 
Њ "џџџџџџџџџ{
&__inference_Drop_1_layer_call_fn_35115Q4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p
Њ "џџџџџџџџџЇ
E__inference_Drop_NXP_2_layer_call_and_return_conditional_losses_35167^4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p 
Њ "&Ђ#

0џџџџџџџџџ
 Ї
E__inference_Drop_NXP_2_layer_call_and_return_conditional_losses_35179^4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p
Њ "&Ђ#

0џџџџџџџџџ
 
*__inference_Drop_NXP_2_layer_call_fn_35157Q4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p 
Њ "џџџџџџџџџ
*__inference_Drop_NXP_2_layer_call_fn_35162Q4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p
Њ "џџџџџџџџџЇ
E__inference_Drop_NXP_3_layer_call_and_return_conditional_losses_35214^4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p 
Њ "&Ђ#

0џџџџџџџџџ
 Ї
E__inference_Drop_NXP_3_layer_call_and_return_conditional_losses_35226^4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p
Њ "&Ђ#

0џџџџџџџџџ
 
*__inference_Drop_NXP_3_layer_call_fn_35204Q4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p 
Њ "џџџџџџџџџ
*__inference_Drop_NXP_3_layer_call_fn_35209Q4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p
Њ "џџџџџџџџџЉ
D__inference_Flatten_1_layer_call_and_return_conditional_losses_35085a7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ		 
Њ "&Ђ#

0џџџџџџџџџ 
 
)__inference_Flatten_1_layer_call_fn_35079T7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ		 
Њ "џџџџџџџџџ ф
A__inference_Pool_1_layer_call_and_return_conditional_losses_32760RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 М
&__inference_Pool_1_layer_call_fn_32766RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџф
A__inference_Pool_2_layer_call_and_return_conditional_losses_32898RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 М
&__inference_Pool_2_layer_call_fn_32904RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџф
A__inference_Pool_3_layer_call_and_return_conditional_losses_32910RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 М
&__inference_Pool_3_layer_call_fn_32916RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџф
A__inference_Pool_4_layer_call_and_return_conditional_losses_33048RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 М
&__inference_Pool_4_layer_call_fn_33054RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџТ
 __inference__wrapped_model_32754(%&1289@ABCNOUVabhipqrsЇЈ>Ђ;
4Ђ1
/,
input_inputџџџџџџџџџии
Њ "1Њ.
,
targets!
targetsџџџџџџџџџД
@__inference_input_layer_call_and_return_conditional_losses_34686p9Ђ6
/Ђ,
*'
inputsџџџџџџџџџии
Њ "/Ђ,
%"
0џџџџџџџџџжж
 
%__inference_input_layer_call_fn_34675c9Ђ6
/Ђ,
*'
inputsџџџџџџџџџии
Њ ""џџџџџџџџџжжх
G__inference_sequential_1_layer_call_and_return_conditional_losses_34090(%&1289@ABCNOUVabhipqrsЇЈFЂC
<Ђ9
/,
input_inputџџџџџџџџџии
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 х
G__inference_sequential_1_layer_call_and_return_conditional_losses_34180(%&1289@ABCNOUVabhipqrsЇЈFЂC
<Ђ9
/,
input_inputџџџџџџџџџии
p

 
Њ "%Ђ"

0џџџџџџџџџ
 р
G__inference_sequential_1_layer_call_and_return_conditional_losses_34520(%&1289@ABCNOUVabhipqrsЇЈAЂ>
7Ђ4
*'
inputsџџџџџџџџџии
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 р
G__inference_sequential_1_layer_call_and_return_conditional_losses_34666(%&1289@ABCNOUVabhipqrsЇЈAЂ>
7Ђ4
*'
inputsџџџџџџџџџии
p

 
Њ "%Ђ"

0џџџџџџџџџ
 Н
,__inference_sequential_1_layer_call_fn_33420(%&1289@ABCNOUVabhipqrsЇЈFЂC
<Ђ9
/,
input_inputџџџџџџџџџии
p 

 
Њ "џџџџџџџџџН
,__inference_sequential_1_layer_call_fn_34000(%&1289@ABCNOUVabhipqrsЇЈFЂC
<Ђ9
/,
input_inputџџџџџџџџџии
p

 
Њ "џџџџџџџџџИ
,__inference_sequential_1_layer_call_fn_34326(%&1289@ABCNOUVabhipqrsЇЈAЂ>
7Ђ4
*'
inputsџџџџџџџџџии
p 

 
Њ "џџџџџџџџџИ
,__inference_sequential_1_layer_call_fn_34395(%&1289@ABCNOUVabhipqrsЇЈAЂ>
7Ђ4
*'
inputsџџџџџџџџџии
p

 
Њ "џџџџџџџџџд
#__inference_signature_wrapper_34257Ќ(%&1289@ABCNOUVabhipqrsЇЈMЂJ
Ђ 
CЊ@
>
input_input/,
input_inputџџџџџџџџџии"1Њ.
,
targets!
targetsџџџџџџџџџЅ
B__inference_targets_layer_call_and_return_conditional_losses_35246_ЇЈ0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 }
'__inference_targets_layer_call_fn_35235RЇЈ0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "џџџџџџџџџ