<style>

  .bar{


  }
</style>

<main>
    <div style:position="relative" style:width={width}px style:height={height}px style:background-color="black">
      <div class="bar" style:background-color="red" style:position="absolute" style:width={barWidth}px style:height={barHeight}px style:left={barXCoordinate}px style:bottom={barYCoordinate}px>

      </div>
      <div style:position="absolute" style:width={canvasSize}px style:height={canvasSize}px style:left={canvasXCoordinate}px style:bottom={canvasYCoordinate}px>
        <P5 {sketch}/>
      </div>
    </div>
  </main>
  
  
  <script>
    export let width;
    export let height;

    let verticalBarWidth = 30;
    let horizontalBarHeight = 30;

    let barWidth;
    let barHeight;
    let canvasSize;

    let canvasXCoordinate;
    let canvasYCoordinate;

    let barXCoordinate;
    let barYCoordinate;

    let bufferWidth = 1000;
    let bufferHeight = 1000;

    $: if(width<height-horizontalBarHeight){ //Vertical mode
        barWidth = width;
        barHeight = horizontalBarHeight;

        canvasSize = Math.min(width, height);

        canvasXCoordinate = 0;
        canvasYCoordinate = (height-canvasSize)/2.;

        barXCoordinate = 0;
        barYCoordinate = (height-canvasSize)/2 - horizontalBarHeight;
    }
    else if (width < height + verticalBarWidth){ //Transition mode
        barWidth = width;
        barHeight = horizontalBarHeight;

        canvasSize = Math.min(width, height-horizontalBarHeight);

        canvasXCoordinate = (width-canvasSize)/2
        canvasYCoordinate = horizontalBarHeight;

        barXCoordinate = 0;
        barYCoordinate = 0;
    }
    else{ //Landscape mode
        barHeight = height;
        barWidth = verticalBarWidth;

        canvasSize = Math.min(width, height);
        
        canvasXCoordinate = (width-canvasSize)/2;
        canvasYCoordinate = 0;

        barXCoordinate = (width - canvasSize)/2 + canvasSize;
        barYCoordinate = 0;
    }


    

    import P5 from 'p5-svelte';
  
  
    var drawingBuffer;
  
    const sketch = (p5) => {
      p5.draw = () => {
        if (p5.mouseIsPressed) {
          drawingBuffer.fill(0);
          drawingBuffer.ellipse(p5.mouseX/canvasSize*bufferWidth, p5.mouseY/canvasSize*bufferHeight, 80, 80);
        } 
  
        p5.image(drawingBuffer, 0, 0, canvasSize, canvasSize);
      };
  
  
      p5.setup = () => {
        p5.createCanvas(canvasSize, canvasSize);
        p5.background(200, 0, 100);
        drawingBuffer = p5.createGraphics(bufferWidth, bufferHeight);
        drawingBuffer.background(200, 0, 100);
      };
  
      
  
      p5.windowResized= () => {
        p5.resizeCanvas(canvasSize, canvasSize, true);
      }
    };
  
  
  </script>
  
  
  
  