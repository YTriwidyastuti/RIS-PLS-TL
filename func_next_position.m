function [x_fin, y_fin, theta_fin] = func_next_position(x_init, y_init, speed, theta, xmax, ymax)

x_pos = x_init + (speed*cos(theta));
y_pos = y_init + (speed*sin(theta));

if x_pos >= xmax
    if y_pos < 0
        x_init = xmax - speed;
        y_init = 0 + speed;
        theta_fin = 3*pi/4;
    else
        if y_pos >= ymax
            x_init = xmax - speed;
            y_init = ymax - speed;
            theta_fin = 5*pi/4;
        else
            ratio = abs((xmax-x_init)/(x_pos-x_init));
            x_init = xmax - speed;
            y_init = ratio*(y_pos-y_init) + y_init;
            theta_fin = pi;
        end
    end
else
    if x_pos <= 0
        if y_pos <= 0
            x_init = 0 + speed;
            y_init = 0 + speed;
            theta_fin = pi/4;
        else
            if y_pos >= ymax
                x_init = 0 + speed;
                y_init = ymax - speed;
                theta_fin = 7*pi/4;
            else
                ratio = abs((0-x_init)/(x_pos-x_init));
                x_init = 0 + speed;
                y_init = ratio*(y_pos-y_init) + y_init;
                theta_fin = 2*pi;
            end
        end
    else
        if y_pos <= 0
            ratio = abs((0-y_init)/(y_pos-y_init));
            x_init = ratio*(x_pos-x_init) + x_init;
            y_init = 0 + speed;
            theta_fin = pi/2;
        else
            if y_pos >= ymax
                ratio = abs((ymax-y_init)/(y_pos-y_init));
                x_init = ratio*(x_pos-x_init) + x_init;
                y_init = ymax - speed;
                theta_fin = 6*pi/4;
            else
                theta_fin = theta;
            end
        end
    end
end

x_fin = x_init + (speed*cos(theta_fin));
y_fin = y_init + (speed*sin(theta_fin));

end