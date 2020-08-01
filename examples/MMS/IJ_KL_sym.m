function r = IJ_KL_sym(a,b)
N = size(a,1);
r = sym(zeros(N,N,N,N));
for i = 1:N
  for j = 1:N
    for k = 1:N
      for l = 1:N
        r(i,j,k,l) = a(i,j)*b(k,l);
      end
    end
  end
end
end