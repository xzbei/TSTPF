% function generate_video(Iend,insertnum)

    savePath = '/Users/beixinzhu/Desktop/movie_result/bag_slow_speed/';

    writerObj = VideoWriter([savePath,'movie.avi']); % Name it.
    writerObj.FrameRate = 3; % How many frames per second - change this to make your video look nice
    open(writerObj);

    for i= 0 : 178
        imname = sprintf([savePath,'frame_%.4d.png'],i);
        I = imread(imname);
        imshow(I);
        drawnow;
        frame = getframe(gcf); writeVideo(writerObj,frame);
    end

    close(writerObj);
