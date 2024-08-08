function [H,U,OBJ] = AROCF(X,Y,m,lmd)
% 初始化变量
[d,N] = size(X);
NC = length(unique(Y));
[~,landmark] = litekmeans(X',m);
[z, ~, ~, ~, ~] = ConstructA_NP(X, landmark',512);
Z = z*z';
D = diag(sum(Z,2));
L = D - Z;
OBJ = [];
H = initializeG(N,NC);
U = initialize(N,NC);
for i=1:10
    % update H
    a = max(max(L)) * lmd;
    A = (ceil(a)+1) * eye(size(L)) - lmd * L;
    B = X' * X * U;
    C = 2 * B + 2 * A * H;
    [UU,TT,WW] = svd(C,'econ');
    H = UU * WW;
    % update U
    U = H;
    obj = norm(X - X * U * H','fro')^2 + lmd * trace(H'*L*H);
    OBJ = [OBJ,obj];
end
end