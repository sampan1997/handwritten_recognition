function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
X=[ones(m,1) X]; %5000*401
a2=zeros(hidden_layer_size,m);
z2=zeros(hidden_layer_size,m);
z2=Theta1*transpose(X);%25*5000
a2=sigmoid(z2);%25*5000
a2=[ones(1,m); a2];%26*5000
%a3=zeros(num_labels,m);
z3=zeros(num_labels,m);%10*5000
z3= Theta2*(a2);
h=transpose(sigmoid(z3));%5000*10
y1=zeros(m,num_labels);%5000*10
for i=1:m
  y1(i,y(i,1))=1;
  end
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

for i=1:m
  for j=1:num_labels
  J=J+ -y1(i,j).*log(h(i,j))-(1-y1(i,j)).*log(1-h(i,j));
 end
 end  
J=J/m;
T=0;
for i=1:hidden_layer_size
  for j=1:input_layer_size
    T=T+Theta1(i,j+1)*Theta1(i,j+1);
  end
end
for i=1:num_labels
  for j=1:hidden_layer_size
    T=T+Theta2(i,j+1)*Theta2(i,j+1);
  end
end
T=T*lambda/(2*m);
J=J+T;


% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
delta3=zeros(size(y1));%5000*10
delta3=h-y1;%5000*10

delta2=zeros(hidden_layer_size,m);%25*5000
delta2=transpose(Theta2)*transpose(delta3);#26*5000
delta2=delta2(2:end,:);
delta2=transpose(delta2.*sigmoidGradient(z2));#5000*25
#until this delta3 is 5000*10 and delta2 is 5000*25
Theta1_grad=Theta1_grad+transpose(delta2)*(X);%25*401
Theta2_grad=Theta2_grad+transpose(delta3)*transpose(a2);%10*26
Theta1_grad=Theta1_grad/m;
Theta2_grad=Theta2_grad/m;
% Part 3: Implement regularization with the cost function and gradients.
for i=1:hidden_layer_size
  for j=2:input_layer_size+1
    Theta1_grad(i,j)=Theta1_grad(i,j)+(lambda*Theta1(i,j)/(m));
  end
end
for i=1:num_labels
  for j=2:hidden_layer_size+1
    Theta2_grad(i,j)=Theta2_grad(i,j)+(lambda*Theta2(i,j)/(m));
  end
  end
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%    

  


















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
