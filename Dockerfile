FROM emscripten/emsdk:2.0.31 as builder

WORKDIR /usr/src/app

COPY . /usr/src/app/

RUN rm -rf build
RUN mkdir build && cd build && \
    emcmake cmake .. && \
    make

FROM nginx:1.25

WORKDIR /usr/share/nginx/html

COPY --from=builder /usr/src/app/build/examples/wasm/index.html .
COPY --from=builder /usr/src/app/build/examples/wasm/libmain.js .
COPY --from=builder /usr/src/app/build/examples/wasm/helpers.js .
COPY --from=builder /usr/src/app/build/examples/wasm/libmain.wasm .
COPY --from=builder /usr/src/app/build/examples/wasm/libmain.worker.js .

RUN rm /etc/nginx/conf.d/default.conf
COPY ./examples/wasm/nginx.conf /etc/nginx/conf.d/

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
