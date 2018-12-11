function q = fdrlasso(tpp, delta, epsi)
%--------------------------------------------------------------------------
% This function calculates the Lasso trade-off curve given tpp (true
% positive proportion), delta = n/p (shape of the design matrix, or
% subsampling rate), and epsi = k/p (sparsity ratio).
% All tpp, delta, and epsi are between 0 and 1; if the
% pair (delta, epsi) is above the Donoho-Tanner phase transition, tpp
% should be no larger than u^\star = powermax(delta, epsi)
%--------------------------------------------------------------------------
% Copyright @ Weijie Su, Malgorzata Bogdan, and Emmanuel Candes, 2015
%--------------------------------------------------------------------------


if tpp > powermax(delta, epsi)
  disp('Invalid input!');
  return;
end

if tpp == 0
  q = 0;
  return
end

%% make stepsize smaller for higher accuracy
stepsize = 0.1;
tmax = max(10, sqrt(delta/epsi/tpp) + 1);
tmin = tmax - stepsize;

while tmin > 0
  if lsandwich(tmin, tpp, delta, epsi) < rsandwich(tmin, tpp)
    break
  end
  tmax = tmin;
  tmin = tmax - stepsize;
end

if tmin <= 0
  stepsize = stepsize/100;
  tmax = max(10, sqrt(delta/epsi/tpp) + 1);
  tmin = tmax - stepsize;
  while tmin > 0
    if lsandwich(tmin, tpp, delta, epsi) < rsandwich(tmin, tpp)
      break
    end  
    tmax = tmin;
    tmin = tmax - stepsize;
  end
end
  
diff = tmax - tmin;
while diff > 1e-6
  tmid = 0.5*tmax + 0.5*tmin;
  if lsandwich(tmid, tpp, delta, epsi) > rsandwich(tmid, tpp)
    tmax = tmid;
  else    
    tmin = tmid;
  end
  diff = tmax - tmin;
end

t = (tmax + tmin)/2;

q = 2*(1-epsi)*normcdf(-t)/(2*(1-epsi)*normcdf(-t) + epsi*tpp);

return;

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function L = lsandwich(t, tpp, delta, epsi)
Lnume = (1-epsi)*(2*(1+t^2)*normcdf(-t) - 2*t*normpdf(t)) + epsi*(1+t^2) - delta;
Ldeno = epsi*((1+t^2)*(1-2*normcdf(-t)) + 2*t*normpdf(t));
L = Lnume/Ldeno;
return;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%
function R = rsandwich(t, tpp)
R = (1 - tpp)/(1 - 2*normcdf(-t));
return;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% highest power for delta < 1 and epsilon > epsilon_phase
function power = powermax(delta, epsilon)
if delta >= 1
  power = 1;
  return;
end
epsilon_star = epsilonDT(delta);
if epsilon <= epsilon_star
  power = 1;
  return;
end
power = (epsilon - epsilon_star)*(delta - epsilon_star)/epsilon/(1 - epsilon_star) + epsilon_star/epsilon;
return
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function epsi = epsilonDT(delta)
minus_f = @(x)-(1+2/delta*x*normpdf(x) - 2/delta*(1+x^2)*normcdf(-x))/(1+x^2-2*(1+x^2)*normcdf(-x)+2*x*normpdf(x))*delta;
alpha_phase = fminbnd(minus_f, 0, 8);
epsi = -feval(minus_f, alpha_phase);
return
end
