a_1=[ones(size(X,1),1) X];
z_2=a_1*Theta1';
a_2=sigmoid(z_2);
a_2=[ones(size(a_2,1),1) a_2];
z_3=a_2*Theta2';
a_3=sigmoid(z_3);

Y= (1:num_labels) == y;

%% Step-2
delta_3= a_3-Y;

%% Step-3

delta_2= delta_3*Theta2(:,2:end).*sigmoidGradient(z_2);


%% Step-4

DEL_2 = delta_3'*a_2;
DEL_1 = delta_2'*a_1;
