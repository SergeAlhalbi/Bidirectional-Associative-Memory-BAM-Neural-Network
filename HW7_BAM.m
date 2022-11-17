%==========================================================================
% Project 7
%==========================================================================

%% Orthogonal Pairs
X_dim = 12;
Y_dim = 20;
Inew_12 = orthoimages_with_dim(X_dim);
Inew_20 = orthoimages_with_dim(Y_dim);
input_neurons = X_dim^2;
output_neurons = Y_dim^2;
X = zeros(input_neurons,10);
Y = zeros(output_neurons,10);
for i = 1:10 
    X(:,i) = reshape(Inew_12(:,:,i),[],1);
    Y(:,i) = reshape(Inew_20(:,:,i+4),[],1);
end

% Initial weights
W = X*Y';
V = Y*X';

% Noise
noise = [0, 0.1, 0.2, 0.3, 0.4, 0.5]; % [0.4, 0.42, 0.44, 0.46, 0.48, 0.5]; Uncomment to check the criticial noise

for h = 1 : length(noise)
    % Adding salt-and-pepper noise
    I = imnoise(X,'salt & pepper' ,noise(h));
    I(I>0.5)=1;
    I(I~=1)= -1;
    
    I_Y = Y;
    I_Y(I_Y>0.5)=1;
    I_Y(I_Y~=1)= -1;
    
    % Plot images from number pattern 1 to 10
    rows = 2;
    columns = 5;
    image = zeros((X_dim+1)*rows,(X_dim+1)*columns);
    image_Y = zeros((Y_dim+1)*rows,(Y_dim+1)*columns);
    task_1_image= ones(X_dim+1,X_dim+1);
    task_1_image_Y= ones(Y_dim+1,Y_dim+1);
    for i = 1:rows
        for j = 1:columns
            n = j + (i-1)*columns;
            task_1_image(1:X_dim,1:X_dim) =reshape(I(:,n),[X_dim,X_dim]);
            task_1_image_Y(1:Y_dim,1:Y_dim) =reshape(I_Y(:,n),[Y_dim,Y_dim]);
            image((X_dim+1)*(i-1)+1:(X_dim+1)*i,(X_dim+1)*(j-1)+1:(X_dim+1)*j) =task_1_image ;
            image_Y((Y_dim+1)*(i-1)+1:(Y_dim+1)*i,(Y_dim+1)*(j-1)+1:(Y_dim+1)*j) =task_1_image_Y ;
        end
    end
    subplot(6,6,6*h-5);
    imshow(image);
    str_train = sprintf('The input image for noise =  %g ',noise(h));
    title(str_train);
    subplot(6,6,6*h-4);
    imshow(image_Y);
    str_train = sprintf('The ground truth image Y');
    title(str_train);
    
%% BAM
    error_bar = ones(1,10);
    error_bar_y = ones(1,10);
    x_out = zeros(X_dim^2,10);
    y_out = zeros(Y_dim^2,10);

    for i = 1: 10


        x = I(:,i);
        x = orth(x);
        y = I_Y(:,i);
        y = orth(y);
        E = 1;
        E_new = 0;
        count = 1;
        iteration= 0;
        change = 1;
        fprintf("Processing number: %d ", i);
                fprintf('\n');
        while(change~=0)

            Net =  W' * x ;
            Net(Net > 0) = 1;
            Net(Net==0)= y(Net==0);
            Net(Net < 0) = -1;
            % Net is y at this step
            Net_x = V' * Net;
            Net_x(Net_x > 0) = 1;
            Net_x(Net_x==0)= x(Net_x==0);
            Net_x(Net_x < 0) = -1;
            
            
            E = V' * W' * Net_x;
            change = E - E_new;
            E_new = E;
            x = Net_x;
            y = Net;

            iteration = iteration +1;
            fprintf(1, repmat('\b',1,count));
            count=fprintf("Current iterations: %d, energy: %f ", iteration, E);
            if iteration > 100
                break
            end
        end
        fprintf('\n')
        x_out(:,i) = Net_x;
        y_out(:,i) = Net;
        error = sum(abs(x_out(:,i) - X(:,i)))/X_dim^2;
        error_y = sum(abs(y_out(:,i) - Y(:,i)))/Y_dim^2;
        error_bar(1,i) = error;
        error_bar_y(1,i) = error_y;
        
        
    end
    
    % Show the bar chart of the percentage error
    task_1_test= ones(X_dim+1,X_dim+1);
    task_1_test_Y= ones(Y_dim+1,Y_dim+1);

    for i = 1:rows
        for j = 1:columns
            n = j + (i-1)*columns;
            task_1_test(1:X_dim,1:X_dim) =reshape(x_out(:,n),[X_dim,X_dim]);
            task_1_test_Y(1:Y_dim,1:Y_dim) =reshape(y_out(:,n),[Y_dim,Y_dim]);
            image_t((X_dim+1)*(i-1)+1:(X_dim+1)*i,(X_dim+1)*(j-1)+1:(X_dim+1)*j) =task_1_test ;
            image_t_Y((Y_dim+1)*(i-1)+1:(Y_dim+1)*i,(Y_dim+1)*(j-1)+1:(Y_dim+1)*j) =task_1_test_Y ;
        end
    end
    
    subplot(6,6,6*h-3);
    imshow(image_t_Y);
    str_test = sprintf('The output image Y');
    title(str_test);
    
    subplot(6,6,6*h-2);
    imshow(image_t);
    str_test = sprintf('The output image X ');
    title(str_test);
        
    subplot(6,6,6*h-1);
    bar (0:9,error_bar,'b')
    str_error = sprintf('X Percentage Error for noise = %g ',noise(h));
    title(str_error);

    subplot(6,6,6*h);
    bar (0:9,error_bar_y,'b')
    str_error_y = sprintf('Y Percentage Error for noise = %g ',noise(h));
    title(str_error_y);
end
%==========================================================================