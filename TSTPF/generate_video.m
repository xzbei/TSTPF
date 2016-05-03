% function generate_video(Iend,insertnum)

    savePath = '/Users/beixinzhu/Desktop/movie_result/TV/';
    imagePath = '/Users/beixinzhu/Desktop/movie_result/TV/';

    writerObj = VideoWriter([savePath,'movie.avi']); % Name it.
    writerObj.FrameRate = 10; % How many frames per second - change this to make your video look nice
    open(writerObj);

    for i= 1:42
        imname = sprintf([imagePath,'frame_%.4d.png'],i);
        fprintf('i = %d\n',i);
        I = imread(imname);
        imshow(I);
        drawnow;
        frame = getframe(gcf); writeVideo(writerObj,frame);
    end

    close(writerObj);
